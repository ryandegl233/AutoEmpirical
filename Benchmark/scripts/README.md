# Benchmark Scripts

_Script inventory updated on 2026-08-03._

## Entry points

| Script | Purpose |
| --- | --- |
| `prepare_ase2022_stage2_filter_baseline.py` | Build the ASE Stage 2 filtering cohort and manifest |
| `run_ase2022_stage2_filter_baseline.py` | Run the ASE single-LLM Stage 2 filter |
| `prepare_ase2022_llm_baseline.py` | Build the ASE Stage 3 labeling cohort |
| `run_ase2022_llm_baseline.py` | Run the ASE single-LLM Stage 3 baseline |
| `prepare_ase2022_camel_mas_baseline.py` | Prepare ASE CAMEL MAS inputs |
| `run_ase2022_camel_mas_baseline.py` | Run ASE CAMEL MAS modes and controls |
| `collect_issta2024_evidence.py` | Collect ISSTA commit evidence used by preparation |
| `prepare_issta2024_bugs_in_pods_baseline.py` | Build repaired ISSTA code-diff cohorts and manifests |
| `run_issta2024_stage2_filter_baseline.py` | Run the ISSTA single-LLM Stage 2 filter |
| `run_issta2024_llm_baseline.py` | Run the ISSTA single-LLM Stage 3 baseline |
| `run_issta2024_camel_mas_baseline.py` | Run ISSTA native or evidence-anchored CAMEL MAS |
| `list_teacher_models.py` | Inspect models exposed by the configured provider |

## Usage

Run any entry point with `--help` before launching a job:

```powershell
python Benchmark/scripts/run_ase2022_camel_mas_baseline.py --help
python Benchmark/scripts/run_issta2024_camel_mas_baseline.py --help
```

Preparation should precede execution so cohort identity, taxonomy, evidence
mode, and prompt inputs are frozen in a manifest. Use resume options for
interrupted network runs and retain validation failures in the audit output.

Provider environment variables and dependency installation are documented in
the [benchmark guide](../README.md).
