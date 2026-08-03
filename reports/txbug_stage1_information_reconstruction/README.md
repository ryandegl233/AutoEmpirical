# ICSE 2024 Transaction Bugs Stage 1 Discussion Reconstruction

_Integrated into `main`; status verified on 2026-08-03._

This reconstruction restores discussion evidence for all 7,775 Stage 1
candidates while preserving identity, membership, and every CSV field other
than `comments`.

## 🔎 Source boundary and version semantics

The author repository releases the final 140-bug set and reports screening
7,775 candidates, but it does not release a frozen discussion snapshot for the
complete pool. Dataset identities remain the cohort anchor. Recovered GitHub,
MariaDB JIRA, MySQL Bugs, PostgreSQL mailing-list, and SQLite Fossil discussion
is `current_unversioned`.

The canonical sidecar is
`../../Dataset/evidence/icse2024_transaction_bugs_stage1_evidence.jsonl`.

## ✅ Coverage

- Evidence records: 7,775.
- `ok`: 6,825.
- `ok_zero_comments`: 948.
- `source_unavailable`: 2.
- Stage 1 rows changed: 4,612, including 4,610 previously blank and two
  truncated MariaDB discussions.

The unavailable records are MySQL Bugs `104833` and `92993`.
`no_comments_in_source` is a verified zero;
`comments_unavailable_in_source` is an unrecovered source and is not zero.

The repaired comments are integrated into unified and per-paper Stage 1. The
source-specific directories under `../txbug_stage1_reconstruction/` are
retained as provenance, not pending work.

## 🧾 Integrity

- Every Stage 1 record has exactly one evidence record.
- Sidecar record IDs and issue URLs are unique.
- The only changed CSV field is `comments`.
- No non-Transaction-Bugs Stage 1 row changed.
- Per-paper and unified Stage 2/3 files are unchanged by this Stage 1 repair.
- The sidecar has no gold-label keys and no high-confidence secret findings.

Machine-readable results are in `audit.json`.
