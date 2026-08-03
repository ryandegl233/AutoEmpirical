# ICSE 2024 Transaction Bugs Stage 1 discussion reconstruction

This reconstruction restores discussion evidence for all 7,775 Stage 1
candidates while preserving record identity, stage membership, and every field
other than `comments`.

## Source boundary and version semantics

The public author repository releases the final 140-bug `TXBug Set.xlsx` and
describes screening 7,775 issues, but it does not release a frozen discussion
snapshot for the complete candidate pool. The Stage 1 candidate identities in
this repository therefore remain the cohort anchor, while recovered discussion
text is classified as `current_unversioned`: it is currently available or
retained public-source evidence, not a claim that every comment existed at the
paper's original collection cutoff.

Evidence was reconstructed from GitHub Issues, MariaDB JIRA, MySQL Bugs,
PostgreSQL mailing-list threads, and SQLite Fossil tickets. The structured
sidecar is
`Dataset/evidence/icse2024_transaction_bugs_stage1_evidence.jsonl`.

## Coverage

- Evidence records: 7,775
- `ok`: 6,825
- `ok_zero_comments`: 948
- `source_unavailable`: 2
- Stage 1 rows changed: 4,612
  - Previously blank discussions completed: 4,610
  - Previously truncated MariaDB discussions completed: 2

`no_comments_in_source` means the retrieved or retained source evidence
explicitly reports zero comments. `comments_unavailable_in_source` means the
discussion could not be recovered and must not be interpreted as zero.

The two unavailable records are:

- `https://bugs.mysql.com/bug.php?id=104833`
- `https://bugs.mysql.com/bug.php?id=92993`

## Integrity

- Every one of the 7,775 Stage 1 records has exactly one evidence record.
- Record IDs and issue URLs are unique within the evidence sidecar.
- The only changed CSV field is `comments`.
- No non-Transaction-Bugs rows changed in the unified Stage 1 file.
- Per-paper and unified Stage 2/3 files are unchanged.
- The evidence sidecar contains no gold-label keys.
- High-confidence secret scan findings: 0.

Machine-readable audit results are in `audit.json`.
