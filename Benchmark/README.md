# AutoEmpirical Benchmark

_Development status reviewed on 2026-09-28._

The benchmark implements study-specific filtering and fixed-taxonomy annotation
with SingleLLM, CAMEL MAS and an adaptive expert workflow. Shared entry points
cover `ase2022`, `issta2024`, `fse2021`, `icse2021`, `icse2022`, `icse2023` and
`icse2024`. The `icse2022` identifier is retained for compatibility; that
performance study appeared at ICSME.

## Prepare inputs

```powershell
python Benchmark/scripts/prepare_paper_benchmarks.py --help
python Benchmark/scripts/prepare_paper_benchmarks.py --output-root Benchmark/inputs/my_preparation
python Benchmark/scripts/prepare_paper_benchmarks.py --output-root Benchmark/inputs/my_preparation --verify-existing
```

Use a new output directory. Each paper normally receives 50 annotated positives
and 50 unselected candidates for filtering; annotation uses the same 50 positives,
independently of filtering predictions. Stage2-only accepted records are excluded
from the negative pool. Preparation saves source bindings, exclusions, snapshots,
prompts, taxonomy and hashes.

The released v2 package is `Benchmark/inputs/seven_papers_v2/<domain>/`.
Original input bytes are preserved; `release_paths.json` supplies repository-relative
source bindings. Run `git lfs pull` first, then verify the release:

```powershell
python Benchmark/scripts/prepare_paper_benchmarks.py --verify-existing
python Benchmark/scripts/seven_papers/verify_prepared_cohorts.py
```

The original manifests retain historical paths as provenance only. Regenerating
inputs does not guarantee the historical ASE cohort or byte-identical files.
Verification identifies two shared source URLs across the paper cohorts; account
for these when defining train/test boundaries.

## Entry points

```powershell
python Benchmark/scripts/run_paper_llm_baseline.py --help
python Benchmark/scripts/run_paper_camel_mas_baseline.py --help
python Benchmark/scripts/run_adaptive_empirical_workflow.py --help
```

The generic baseline runners accept `--domain` and `--prepared-root`. Choose the
provider and exact model explicitly, and use a new output directory. Generic
baseline outputs default to `Benchmark/runs/seven_papers/<domain>/`. See the
adaptive CLI for its cohort, split, architecture, evidence and execution-profile
options; do not assume all runners share the same preparation flags.

Existing ASE and ISSTA entry points remain available for historical experiments.
Generic MAS runs save reasons, citations and request traces. `--allow-invalid`
retains unsuccessful records and continues execution; it does not turn invalid
outputs into valid predictions.

## Provider and environment setup

Install `requirements.txt` from the repository root; CAMEL-based runs also need
`Benchmark/requirements-mas.txt`. For the complete test environment install
`requirements-dev.txt` from the root. Python 3.13 was used for release validation.

| Provider path | Configuration |
| --- | --- |
| Official Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY`; select `--provider gemini` and an explicit `--model` |
| OpenAI-compatible proxy | `SELF_BASE_URL`, `BASE_URL` or `LLM_BASE_URL`; `SELF_API`, `OPENAI_API_KEY` or `API_KEY` |
| Direct DeepSeek | `DEEPSEEK_API_KEY` or `DEEPSEEK_API`; optional `DEEPSEEK_BASE_URL` |

These names describe this repository's configuration interface. Credential files
are local. Relay continuation tools have separate credential variables and
transport provenance; requested model names do not establish upstream equivalence.

## Evaluation protocol

| Domain/property | Evaluation rule |
| --- | --- |
| Filtering | Agreement with the paper's selection target on the fixed cohort |
| IoT symptoms | Label-set exact match and micro F1; serialized with ` || ` |
| UAV symptoms | Free text; no symptom or joint classification accuracy |
| DL performance symptoms | Constant description; no symptom or joint classification accuracy |
| PyTorch filtering | Author-selection agreement, not proven nonbug detection |
| Invalid/unresolved | Retain in fixed denominators and report separately |

- Keep gold labels, author annotation summaries and later-stage answers out of
  model inputs. Bind prepared inputs and results to explicit versions and hashes.
- Keep invalid and unresolved outputs visible in fixed-cohort denominators.
  Unknown can be a legitimate taxonomy label and differs from invalid output.
- Group shared source URLs across training/test boundaries, including overlap
  across papers. Previously inspected development cases are not unseen holdouts.
- Preserve missing-source markers in provenance while excluding them from
  technical evidence. Reconstructed sources may be `current_unversioned`; do not
  describe them as exact historical snapshots without supporting evidence.
- Separate cohorts, policy versions, providers, retries and continuation batches.
  Do not pool exploratory subsets or mixed-provider outputs into a final result.

Evidence readiness or citation validation does not certify a model's causal
interpretation.

## Local experiment outputs

This update publishes implementations, tests, configurations and selected input
data. Real model predictions, generated explanations, scores, request/response
traces and experiment logs remain local; they are not uploaded as repository
files or release attachments. Use `Benchmark/results/` or `Benchmark/runs/` for
local output, and keep publishable frozen inputs under `Benchmark/inputs/`.

Scoring and verification scripts are part of the implementation. Supply locally
generated predictions when running them. Features that use a historical baseline
use your own local artifact or a newly generated baseline. Explicitly register
its digest using `prepare_baseline_trust_root.py --activate-local`; the manifest
and registry stay under ignored `Benchmark/cache/`. See
[local baseline registration](../docs/methods/local-baseline.md).

## Evidence interventions and experiment two

The adaptive code includes formal/rule-based evidence comparisons, evidence-chain
checking, supplemental snapshots and source-graph experiments. These are
experimental policies on inspected development data.

Experiment two compares E00 (corrected rules-v4 baseline), E10 (additional
grounding and checks), E01 (source graph) and E11 (both). The case-isolated entry is
`tools/run_experiment_two_sharded.py`. It now consumes
`Benchmark/inputs/ase2022_dev48/experiment_two.json`, with no old run manifest
required. This input configuration binds the 48 evidence-only cases, two reserved
method examples, supplemental source text and images. Gold is loaded only by the
offline scorer; it is not passed to inference workers.

```powershell
python tools/run_experiment_two_sharded.py --batch-id my-offline-check
```

The default performs offline validation. `--run` explicitly enables model calls;
`--resume` retains completed records, including invalid outputs. Use separate
batch IDs for changed code, inputs or providers. Additional tools handle UTF-8,
continuation and explicitly configured relay transports. Outputs stay under
`Benchmark/runs/experiment_two/`. See [method notes](../docs/methods/README.md).

External capture tools support page, issue, code and image reads and offline replay;
see [external evidence tools](./EXTERNAL_EVIDENCE_TOOLS.md). See the
[experiment notes](./EXPERIMENT_TWO.md) for protocol details.

## Reproduction contract

Every local run should identify its cohort, record order, task semantics,
taxonomy/prompt versions, evidence snapshot, source version, model/provider,
decoding settings, retries, validity policy and metric denominator. Keep
predictions, audits and cost/latency records locally where available. Model output
artifacts are excluded from this release.

Run `python -m pytest tests -q` for current tests. Avoid collecting tests recursively
from reports containing archived source trees. Pre-cache tokenizer data for fully
offline SDK integration tests: `python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"`.
The cache download needs network access once; tests use synthetic model responses.
