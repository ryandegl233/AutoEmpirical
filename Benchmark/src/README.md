# Benchmark Source Modules

_Module inventory updated on 2026-08-03._

## Modules

| Module | Responsibility |
| --- | --- |
| `ase2022_stage2_filter_baseline.py` | ASE Stage 2 cohort construction, prompt preparation, parsing, validation, and scoring |
| `ase2022_llm_baseline.py` | ASE Stage 3 taxonomy prompt and result-processing logic |
| `ase2022_camel_mas_baseline.py` | ASE CAMEL society, evidence anchoring, finalization, and audit logic |
| `issta2024_bugs_in_pods_baseline.py` | ISSTA commit-diff preparation, task prompts, validation, and metrics |
| `issta2024_evidence_collection.py` | ISSTA commit evidence retrieval and normalization |

## Design boundary

Files in this directory are reusable implementation modules; executable
entry points live in `../scripts/`, and committed run artifacts live in
`../results/`. The implemented code currently covers ASE 2022 and ISSTA 2024,
not all seven dataset domains.

When extending a module, preserve deterministic cohort identity, explicit
evidence modes, schema validation, invalid-output accounting, and separation
between model inputs and gold labels.
