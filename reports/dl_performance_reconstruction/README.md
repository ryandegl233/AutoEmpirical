# DL Performance Evidence Reconstruction

_Integrated into `main`; status verified on 2026-08-03._

This directory documents the evidence reconstruction for
`icse2022_an_empirical_study_on_performance`. It preserves the 6,835 / 2,265 /
136 stage cohorts and changes no membership, `record_id`, `issue_url`, or gold
label.

## Evidence sources

`source/` contains six author-artifact CSVs. Its manifest pins the author
repository commit and records each file's SHA256, byte size, row count, and
schema. `raw/issues.jsonl` contains current GitHub issue responses;
`raw/commits.jsonl` contains normalized author-identified fixing commits.
Unneeded raw GitHub email fields are not included in the committed package.

The author artifact anchors cohort membership and historical comment counts.
Issue text, current comments, commit messages, and patches are
`current_unversioned`, not exact February 2021 snapshots.

## Outputs

- `../../Dataset/evidence/icse2022_dl_performance_evidence.jsonl`: one evidence
  object per Stage 1 record.
- `preview/`: validated per-paper rows used by the guarded integration.
- `audit/integration_audit.csv`: issue, comment, and commit status for stage
  rows.
- `audit/evidence_coverage.csv`: PyTorch, TensorFlow, and overall coverage.
- `audit/comment_count_mismatches.csv`: current comment counts below the author
  snapshot count.
- `audit/missing_author_commits.csv`: Stage 3 rows absent from the author's
  fixing-commit tables.
- `audit/reconstruction_manifest.json`: input hashes, output hashes,
  provenance policy, and completion counts.

## Integrated coverage

- Evidence records: 6,835 / 6,835.
- Current issue responses unavailable with stable 404: 6.
- Unique author-referenced fixing commits recovered: 173 / 173.
- Changed-file entries retained: 710, including 709 patches and one
  source-unavailable patch.
- Five unique issues currently expose fewer comments than the author snapshot.
- One Stage 3 row is absent from the author fixing-commit tables.

These limitations are explicit provenance states, not silently filled fields.

## Evidence modes

- `issue_only`: title and body.
- `issue_discussion`: title, body, and retained comments.
- `full_fix_evidence`: issue discussion plus fixing commits.

Gold labels remain in Stage 3 and are not copied into the evidence sidecar.
