# AutoEmpirical Benchmark

_Benchmark documentation updated on 2026-08-03._

This directory contains executable data preparation, single-LLM, and CAMEL
multi-agent experiments. The currently committed implementations cover ASE
2022 Faults of DL Systems and ISSTA 2024 BugsInPy; the other five dataset
domains do not yet have equivalent benchmark runners.

## 🎯 Tasks

| Task | Model input | Evaluation target |
| --- | --- | --- |
| Stage 2 filtering | Stage 1 information fields | Membership in the paper's Stage 2 cohort |
| Stage 3 labeling | Permitted Stage 2 evidence fields | Gold Stage 3 taxonomy |

Gold membership and taxonomy fields must remain evaluation-only. Preparation
scripts create task-specific cohorts and manifests so later-stage answers are
not copied into prompts.

## 🧪 Implemented experiment families

| Paper | Single LLM | CAMEL MAS | Committed result families |
| --- | --- | --- | --- |
| ASE 2022 | Stage 2 and Stage 3 | Native/evidence-anchored society and finalizer variants | Five-model Stage 2/3 samples, single-LLM controls, finalizer runs |
| ISSTA 2024 | Stage 2 and Stage 3 | Native and evidence-anchored society | Repaired full-code-diff cohorts, single-LLM control, MAS native, MAS evidence-enriched |

The repository does not claim full seven-paper benchmark coverage yet.

## 📁 Layout

| Path | Contents |
| --- | --- |
| `scripts/` | Preparation, execution, evidence collection, and model-listing entry points |
| `src/` | Reusable normalization, prompt, parsing, validation, and MAS logic |
| `results/` | Committed manifests, predictions, audits, metrics, and selected run outputs |
| `configs/` | Reserved for reusable declarative experiment configurations |
| `requirements-mas.txt` | CAMEL and Pydantic dependency constraints |

See the README in each subdirectory for an exact inventory.

## ⚡ Representative commands

Prepare and run the ASE 2022 single-LLM baselines:

```powershell
python Benchmark/scripts/prepare_ase2022_stage2_filter_baseline.py --help
python Benchmark/scripts/run_ase2022_stage2_filter_baseline.py --help
python Benchmark/scripts/prepare_ase2022_llm_baseline.py --help
python Benchmark/scripts/run_ase2022_llm_baseline.py --help
```

Prepare and run the ISSTA 2024 full-code-diff baselines:

```powershell
python Benchmark/scripts/prepare_issta2024_bugs_in_pods_baseline.py --help
python Benchmark/scripts/run_issta2024_stage2_filter_baseline.py --help
python Benchmark/scripts/run_issta2024_llm_baseline.py --help
```

Run the CAMEL multi-agent entry points:

```powershell
python -m pip install -r Benchmark/requirements-mas.txt
python Benchmark/scripts/run_ase2022_camel_mas_baseline.py --help
python Benchmark/scripts/run_issta2024_camel_mas_baseline.py --help
```

Use `--help` as the source of truth for model, stage, cohort, output, retry,
resume, and validation options.

## 🔐 Provider configuration

Do not store credentials in configs, manifests, or result files.

For the OpenAI-compatible proxy path, runners read the base URL from
`SELF_BASE_URL`, `BASE_URL`, or `LLM_BASE_URL`, and the key from `SELF_API`,
`OPENAI_API_KEY`, or `API_KEY`.

For direct DeepSeek CAMEL runs, use `DEEPSEEK_API_KEY` (or
`DEEPSEEK_API`) and optionally `DEEPSEEK_BASE_URL`.

Example for the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "<temporary-key>"
$env:LLM_BASE_URL = "https://example.invalid/v1"
```

## 📏 Output and audit expectations

Each committed experiment family should retain enough information to reproduce
and audit the run:

- preparation manifest and cohort identity;
- prompt/taxonomy version or hash;
- model/provider and decoding parameters;
- raw or normalized predictions;
- schema-validation and invalid-output status;
- aggregate and per-label metrics where applicable;
- evidence mode, cost, and latency metadata when available.

Invalid model outputs must remain visible in audits. Do not silently drop,
repair, or score them as valid predictions.

## 📚 Related documentation

- [Scripts](./scripts/README.md)
- [Reusable source modules](./src/README.md)
- [Committed results](./results/README.md)
- [Dataset guide](../Dataset/README.md)
- [Baseline research plan](../research/baseline_research_plan.md)
