# DL Performance Evidence Reconstruction

This directory documents the evidence reconstruction for
`icse2022_an_empirical_study_on_performance`.

The reconstruction preserves the existing benchmark cohorts:

- Stage 1: 6,835 records
- Stage 2: 2,265 records
- Stage 3: 136 records

No record membership, `record_id`, `issue_url`, or gold-label field is changed.

## Evidence sources

`source/` contains the six author-artifact CSV files used by this reconstruction.
`source/source_manifest.json` pins the author repository commit and records the
SHA256, byte size, row count, and schema of each source file.

`raw/issues.jsonl` contains the current GitHub issue responses used for
normalization. `raw/commits.jsonl` contains normalized author-identified fixing
commit responses. Raw commit payloads are deliberately omitted from the
committed package because GitHub includes author email addresses that are not
needed by the benchmark.

The author artifact anchors cohort membership and historical comment counts.
Issue titles, bodies, comment text, commit messages, and patches were reconstructed
from the current GitHub API and are marked `current_unversioned`; they must not be
described as exact February 2021 snapshots.

## Outputs

- `Dataset/evidence/icse2022_dl_performance_evidence.jsonl` stores one evidence
  object per Stage 1 `record_id`.
- `preview/` stores the validated per-paper rows used by the guarded apply step.
- `audit/integration_audit.csv` records the issue/comment/commit status of every
  stage row.
- `audit/evidence_coverage.csv` reports PyTorch, TensorFlow, and overall coverage.
- `audit/comment_count_mismatches.csv` identifies cases where currently visible
  comments are fewer than the author snapshot count.
- `audit/missing_author_commits.csv` records Stage 3 rows absent from the author's
  fixing-commit tables.
- `audit/reconstruction_manifest.json` records input hashes, preview hashes,
  provenance policy, and completion counts.

## Evidence modes

- `issue_only`: use `title` and `body`.
- `issue_discussion`: add the retained comments stored in `comments` or the
  structured `issue.retained_comments` sidecar field.
- `full_fix_evidence`: additionally use `fixing_commits` from the sidecar.

Gold labels remain in Stage 3 CSV fields and are not copied into the sidecar.
