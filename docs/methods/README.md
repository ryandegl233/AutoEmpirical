# Adaptive benchmark methods

The workflow separates source evidence from taxonomy labels, forms independent
analyses, verifies cited evidence and resolves disagreements. It records validity,
evidence use and execution policy rather than silently dropping unresolved cases.

The released code includes three development protocols:

| Protocol | Entry | Purpose |
| --- | --- | --- |
| SingleLLM / CAMEL / adaptive | `Benchmark/scripts/run_paper_*`, `run_adaptive_empirical_workflow.py` | Native paper filtering and annotation |
| A/B comparison | `Benchmark/scripts/ase2022_dev48/run_four_arms.ps1` | Formal source-grounded examples and evidence-chain checking |
| Experiment two | `tools/run_experiment_two_sharded.py` | E00 rules-v4; E10 grounding/checks; E01 source graph; E11 both |

For A/B, S0 enables neither module; A adds formal examples/comparison; B adds
chain checking; AB enables both. These policies differ from experiment two and
must not be pooled. Two source-only examples are excluded from the 48 development
cases. Source graphs contain explicit source relations, not inferred causal truth.

See the [benchmark guide](../../Benchmark/README.md) for input/evaluation boundaries,
[experiment two](../../Benchmark/EXPERIMENT_TWO.md) for execution, and
[local baseline registration](local-baseline.md) for baseline-preservation mode.
Model predictions, explanations, metrics, result audits and traces remain local.
