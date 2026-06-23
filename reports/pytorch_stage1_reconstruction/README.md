# PyTorch Stage 1 Reconstruction

Generated on 2026-06-22.

This directory contains a first-pass reconstruction of the Stage 1 candidate issue set for:

```text
icse2023_an_empirical_study_on_bugs
```

## Paper Rule Implemented

The PyTorch paper describes its automatic candidate filtering as:

- PyTorch GitHub repository issues
- closed issues
- labeled `triaged`
- having a linked pull request
- as of 2022-10-20

The current reconstruction uses GitHub Search API with:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-10-20
```

The script splits the query by `created:` date windows to avoid GitHub Search API's 1,000-result paging limit.

## Outputs

| File | Meaning |
| --- | --- |
| `pytorch_stage1_candidates.csv` | Normalized candidate issue table |
| `pytorch_stage1_candidates.raw.jsonl` | Raw GitHub Search API issue payloads |
| `fetch_manifest.json` | Query, date windows, and reported counts |

## Current Result

| Item | Count |
| --- | ---: |
| Paper-reported Stage 1 candidates | 2,205 |
| Current GitHub API reconstruction | 2,676 |
| Difference | +471 |
| Local labeled PyTorch records covered | 192 / 194 |

Two local labeled records were not returned by the current GitHub query:

| Issue | Current GitHub state observed during audit |
| --- | --- |
| `https://github.com/pytorch/pytorch/issues/48841` | Closed in 2023, after the 2022-10-20 cutoff |
| `https://github.com/pytorch/pytorch/issues/39007` | Currently open and has no `pull_request` object in the issue payload |

This indicates that current GitHub metadata is not an exact historical snapshot of the paper's data-collection state. The reconstructed set is therefore a close methodological replay, not a byte-for-byte recovery of the authors' original Stage 1 artifact.

## Reproduction Command

```powershell
python scripts\fetch_pytorch_stage1_candidates.py
```
