# ASE2022 Stage 1 to Stage 2 Single-LLM Baseline Design

## Objective

Build a reproducible single-LLM baseline that predicts whether an ASE2022 Stage 1 candidate should be retained in Stage 2. The task is study-specific fault relevance classification, not general-purpose bug detection.

## Ground Truth

The benchmark uses the current AutoEmpirical datasets:

- Stage 1 contains 4,184 ASE2022 candidate records.
- Stage 2 contains 683 retained records.
- A Stage 1 record is `accepted_fault` when its `record_id` occurs in Stage 2.
- All other Stage 1 records are `rejected_candidate`.

The loader must verify that every ASE2022 Stage 2 `record_id` occurs exactly once in Stage 1. It must fail with a clear error if this containment or uniqueness invariant is violated.

This membership label approximates the paper's manual screening decision. It does not reconstruct the paper's intermediate set of 1,293 automatically filtered issues because that membership is not preserved in the repository.

## Leakage Controls

The model may receive only these Stage 1 fields:

- `title`
- `body`
- `comments`
- `state`
- `created_at`

The prompt and serialized model request must not expose downstream labels or provenance fields, including `symptom`, `root_cause`, `bug_type`, `component`, `sub_component`, `fix_type`, `source_file`, `source_sheet`, `original_label_json`, or Stage 2 membership.

Ground-truth labels may be stored in local sample and prediction artifacts for evaluation, but must remain outside the messages sent to the model.

## Sampling

The preparation command produces a deterministic balanced pilot sample:

- 50 `accepted_fault` records.
- 50 `rejected_candidate` records.
- A fixed seed recorded in the artifact metadata.
- Records without usable title, body, and comments are excluded.

The runner must also support the full 4,184-record dataset without requiring code changes. Balanced-pilot metrics and full-distribution metrics must not be presented as interchangeable.

## Prompt Contract

The prompt describes the ASE2022 screening target and instructs the model to retain a record only when the available issue evidence describes a real, identifiable fault in a JavaScript-based deep-learning system.

Examples of rejection criteria include feature requests, usage questions without evidence of a system fault, unclear or insufficient descriptions, irrelevant discussions, and keyword matches that are not errors.

The model must return strict JSON with exactly one decision label:

```json
{"decision": "accepted_fault"}
```

The only allowed values are `accepted_fault` and `rejected_candidate`. Additional prose, malformed JSON, missing decisions, and unknown labels are invalid outputs.

## Architecture

Stage 2 filtering will use a focused module and CLI rather than further expanding the Stage 3 labeling module:

- `Benchmark/src/ase2022_stage2_filter_baseline.py` owns dataset loading, membership labeling, leakage-safe prompt construction, stratified sampling, parsing, evaluation, and orchestration.
- `Benchmark/scripts/prepare_ase2022_stage2_filter_baseline.py` creates deterministic sample and prompt artifacts.
- `Benchmark/scripts/run_ase2022_stage2_filter_baseline.py` configures models and API transport, then runs or resumes predictions.
- `tests/test_ase2022_stage2_filter_baseline.py` verifies the domain logic.
- `tests/test_run_ase2022_stage2_filter_baseline.py` verifies CLI configuration and runner integration.

Existing Stage 3 HTTP transport behavior may be reused through explicitly imported helpers where that avoids duplication, but Stage 2 filtering must expose its own task-specific interfaces and artifacts.

## Artifacts

The default output directory is:

`Benchmark/results/ase2022_stage2_filter_baseline/`

Preparation creates:

- A balanced sample CSV containing identifiers, safe evidence fields, and ground truth.
- An anchored-prompts JSONL artifact.
- A manifest JSON recording counts, seed, sampling rules, source paths, and label semantics.

Each model run creates:

- A prediction JSONL file with record ID, ground truth, raw response, parsed decision, validity, error details, and retry metadata.
- A metrics JSON file with run configuration and evaluation results.

Resume behavior keys completed work by `record_id`. Duplicate or conflicting saved predictions must produce an explicit error rather than silently selecting one.

## Evaluation

Metrics treat `accepted_fault` as the positive class and include:

- Accuracy
- Precision
- Recall
- F1
- Specificity
- Balanced accuracy
- True positives, false positives, true negatives, and false negatives
- Valid and invalid output counts
- Invalid-output rate

Classification metrics are computed over valid predictions. The metrics artifact must state the valid denominator so invalid responses cannot be mistaken for negative predictions.

## Error Handling

The implementation must fail clearly for:

- Missing or duplicate Stage 1/Stage 2 record IDs
- Stage 2 records absent from Stage 1
- Unsupported labels
- Empty eligible sampling pools
- Requested class sample sizes larger than their eligible pools
- Duplicate resume records
- Missing API configuration

Transient model or transport failures use the existing bounded retry behavior. Exhausted calls are persisted as invalid predictions so the run remains auditable and resumable.

## Testing and Verification

Implementation follows test-driven development. Tests must demonstrate:

- Correct membership-derived labels and class counts
- Failure on broken lineage or duplicate IDs
- No forbidden field appears in a model prompt
- Deterministic 50/50 stratified sampling
- Strict acceptance and rejection parsing
- Invalid handling for malformed or unsupported outputs
- Correct confusion matrix and derived metrics
- Resume without duplicate model calls
- CLI defaults and explicit overrides

Completion requires the focused tests, the existing Stage 3 baseline tests, and the full repository test suite to pass. A local preparation smoke test must also generate a 100-record sample with exactly 50 records per class.

## Documentation Language

Repository documentation and reports should call the task "ASE2022 Stage 2 filtering" or "study-specific fault relevance classification." Any shorter phrase such as "bug detection" must be accompanied by the operational definition that positive means membership in the paper-derived Stage 2 dataset.
