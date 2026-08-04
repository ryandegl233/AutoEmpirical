# PyTorch Stage 1 Convergence Experiments

_Generated on 2026-06-23; selected convergence set integrated into `main` and
status updated on 2026-08-03._

These experiments reconcile the paper's 2,205 Stage 1 candidates with current
GitHub Search state.

## Baseline reconstruction

The paper-described current-metadata query returns 2,676 records and covers
192/194 local labeled issues:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-10-20
```

The missing local records are issues `39007` and `48841`, whose current
metadata differs from the paper-era state.

## Label-exclusion experiment

Excluding `module: docs` and `enhancement` yields 2,252 candidates but covers
only 179/194 labeled records. It is numerically close but not suitable for
integration. The broader search is stored in
`offline_label_exclusion_search.csv`.

## Exact-count cutoff experiment

Changing only the current `closed` cutoff yields:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-03-09
```

| Variant | Count | Difference from 2,205 | Labeled coverage |
| --- | ---: | ---: | ---: |
| `closed<=2022-03-09` | 2,205 | 0 | 192 / 194 |

This date conflicts with the paper's stated 2022-10-20 cutoff and may reflect
snapshot drift, collection lag, or numerical coincidence.

## Generated files

| File | Meaning |
| --- | --- |
| `no_docs_no_enhancement.csv` | Closest label-exclusion candidate set |
| `no_docs_no_enhancement.raw.jsonl` | Raw payloads for that variant |
| `no_docs_no_enhancement_manifest.json` | Variant manifest |
| `closed_cutoff_2022_03_09_exact_2205.csv` | Selected exact-count set |
| `closed_cutoff_2022_03_09_exact_2205_manifest.json` | Exact-count manifest |
| `offline_label_exclusion_search.csv` | Offline exclusion search |

## Integration outcome

The exact-count set replaced the placeholder reconstruction. Two local Stage 2
records absent from the query are retained as lineage supplements, producing
the dataset's 2,207-row PyTorch Stage 1 cohort. This is documented as a
convergence reconstruction, not the authors' original artifact.
