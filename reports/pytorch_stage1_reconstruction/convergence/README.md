# PyTorch Stage 1 Convergence Experiments

Generated on 2026-06-23.

These experiments try to reconcile the PyTorch paper's reported Stage 1 candidate count, 2,205, with the current GitHub Search API state.

## Baseline Reconstruction

The first-pass reconstruction used the paper-described automatic filters as closely as possible:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-10-20
```

Result:

| Variant | Count | Difference from 2,205 | Local labeled coverage |
| --- | ---: | ---: | ---: |
| baseline `closed<=2022-10-20` | 2,676 | +471 | 192 / 194 |

The two local labeled records missing from the baseline are:

- `https://github.com/pytorch/pytorch/issues/39007`
- `https://github.com/pytorch/pytorch/issues/48841`

These appear to be caused by current GitHub metadata differing from the paper-era snapshot.

## Label-Exclusion Experiment

The closest simple label-exclusion rule found in the first pass was:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-10-20 -label:"module: docs" -label:enhancement
```

Result:

| Variant | Count | Difference from 2,205 | Local labeled coverage |
| --- | ---: | ---: | ---: |
| exclude `module: docs` and `enhancement` | 2,252 | +47 | 179 / 194 |

Although this gets close numerically, it removes 15 local labeled records. It is therefore not a good reconstruction rule.

The broader offline label-exclusion search is saved in:

```text
offline_label_exclusion_search.csv
```

## Moving Closed-Date Cutoff Experiment

Using the current GitHub metadata, changing only the `closed` cutoff gives an exact count match:

```text
repo:pytorch/pytorch is:issue is:closed label:triaged linked:pr closed:<=2022-03-09
```

Result:

| Variant | Count | Difference from 2,205 | Local labeled coverage |
| --- | ---: | ---: | ---: |
| `closed<=2022-03-09` | 2,205 | 0 | 192 / 194 |

The covered local labeled records have current GitHub `closed_at` values ranging from 2019-05-11 to 2022-02-10, so this earlier cutoff does not remove any local labeled record that was covered by the baseline reconstruction.

However, this should be treated cautiously: the paper text says "As of 20 October 2022", while this convergence point is 2022-03-09. The match may reflect a historical snapshot mismatch, paper-data collection lag, or a numerical coincidence.

## Generated Files

| File | Meaning |
| --- | --- |
| `no_docs_no_enhancement.csv` | Full candidate list for the closest label-exclusion variant |
| `no_docs_no_enhancement.raw.jsonl` | Raw GitHub payloads for that variant |
| `no_docs_no_enhancement_manifest.json` | Manifest for that variant |
| `closed_cutoff_2022_03_09_exact_2205.csv` | Exact-count candidate list derived from the baseline reconstruction |
| `closed_cutoff_2022_03_09_exact_2205_manifest.json` | Manifest for the exact-count variant |
| `offline_label_exclusion_search.csv` | Offline search over label-exclusion combinations |

## Current Recommendation

For replacing the 2,011 PyTorch placeholder rows, the best current candidate is:

```text
closed_cutoff_2022_03_09_exact_2205.csv
```

It exactly matches the paper's Stage 1 count and preserves the same labeled-record coverage as the broader baseline query. But because its cutoff date conflicts with the paper's stated 2022-10-20 date, it should be documented as a convergence reconstruction rather than as the authors' original artifact.
