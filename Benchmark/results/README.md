# Benchmark Results

_Result inventory updated on 2026-08-03._

This directory contains the selected experiment artifacts retained for audit
and reproduction. It is not a claim that every paper or every planned baseline
has been run.

## Committed result families

| Path | Contents |
| --- | --- |
| `ase2022_llm_baseline/` | Five-model, 50-record Stage 3 single-LLM sample |
| `ase2022_stage2_filter_baseline/` | Five-model, 100-record balanced Stage 2 filtering sample |
| `ase2022_camel_mas_baseline/full_evidence_anchored_deepseek/` | Evidence-anchored MAS outputs and matching single-LLM control |
| `ase2022_camel_mas_baseline/full_society_finalizer_deepseek/` | Finalizer-based ASE MAS run and required artifacts |
| `issta2024_bugs_in_pods_baseline/code_diff_repaired/` | Repaired full-code-diff cohorts, manifests, and preparation audit |
| `issta2024_bugs_in_pods_baseline/full_code_diff_deepseek_v4_flash/` | ISSTA single-LLM control, MAS native, and MAS evidence-anchored outputs |
| `ase2022_baseline_results_summary.md` | Human-readable ASE result summary |

## Interpretation rules

- Read each run manifest before comparing metrics; sample, model, prompt,
  evidence mode, and validation policy can differ.
- Keep invalid and abstained outputs in denominators unless a report explicitly
  defines another policy.
- Treat `single_llm_control`, `mas_native`, and evidence-anchored/finalizer
  directories as distinct experiment conditions.
- ISSTA results in `full_code_diff_deepseek_v4_flash/` use the repaired
  full-code-diff input, not the earlier incomplete dataset.
- Older smoke outputs and runs produced before the corresponding data repair
  are intentionally not part of the retained result set.

## Adding results

Retain the cohort or record IDs, preparation manifest, prompt/taxonomy
provenance, model parameters, prediction output, validation audit, and metrics.
Do not commit credentials or transient caches.

See the [benchmark guide](../README.md) for execution entry points.
