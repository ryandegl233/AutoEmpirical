# PyTorch Stage 1 Reconstruction

_Generated on 2026-06-22; selected reconstruction integrated into `main` and
status updated on 2026-08-03._

This directory preserves a broad methodological replay of the Stage 1 candidate
issue set for `icse2023_an_empirical_study_on_bugs`.

## Paper rule implemented

The paper describes closed PyTorch GitHub issues labeled `triaged`, having a
linked pull request, as of 2022-10-20. The broad replay uses:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-10-20
```

Date-window subqueries avoid GitHub Search API's 1,000-result paging limit.

## Outputs

| File | Meaning |
| --- | --- |
| `pytorch_stage1_candidates.csv` | Normalized broad-query candidates |
| `pytorch_stage1_candidates.raw.jsonl` | Raw GitHub Search issue payloads |
| `fetch_manifest.json` | Query windows and reported counts |

## Broad-query result

| Item | Count |
| --- | ---: |
| Paper-reported Stage 1 candidates | 2,205 |
| Current broad-query reconstruction | 2,676 |
| Difference | +471 |
| Local labeled PyTorch records covered | 192 / 194 |

The two uncovered local labeled records are issue `48841`, currently closed
after the cutoff, and issue `39007`, currently open without a pull-request
object. Current GitHub metadata is therefore not an exact historical snapshot.

## Integration outcome

The dataset uses the exact-count 2,205-row convergence set documented in
[`convergence/README.md`](./convergence/README.md), plus two explicit Stage 2
lineage supplements for the uncovered labeled records. The integrated PyTorch
Stage 1 cohort contains 2,207 rows.

The exact-count cutoff conflicts with the paper's stated date. It is a
transparent convergence reconstruction, not a claim to be the authors' frozen
artifact. Exact historical provenance requires that original snapshot.

## Reproduction command

```powershell
python scripts\fetch_pytorch_stage1_candidates.py
```
