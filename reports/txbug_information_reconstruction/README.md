# ICSE 2024 Transaction Bugs Discussion Reconstruction

_Integrated into `main`; status verified on 2026-08-03._

This reconstruction restores public discussion for the 140-record Stage 2/3
cohort while preserving identity, membership, and gold labels. The full
7,775-record Stage 1 repair is documented in
[`../txbug_stage1_information_reconstruction/README.md`](../txbug_stage1_information_reconstruction/README.md).

## Version semantics

The author artifact does not contain a frozen copy of every source discussion.
Recovered text is `current_unversioned`: currently visible or retained public
evidence, not a claim that every comment existed at the original cutoff.

## Source coverage

| Source type | Records | With discussion |
| --- | ---: | ---: |
| GitHub issue | 71 | 68 |
| MariaDB JIRA | 24 | 19 |
| MySQL Bugs | 33 | 31 |
| PostgreSQL mail thread | 6 | 6 |
| SQLite Fossil ticket | 6 | 3 |

## Retrieval status

- `ok`: 127.
- `ok_zero_comments`: 11.
- `source_unavailable`: 2.

`no_comments_in_source` means authoritative evidence exposes zero comments.
`comments_unavailable_in_source` means evidence could not be recovered and is
not a verified zero.

The reconstruction is integrated into unified and per-paper Stage 2/3 files.
The canonical sidecar is
`../../Dataset/evidence/icse2024_transaction_bugs_evidence.jsonl`.

## Integrity

- Evidence records: 140.
- Record IDs and issue URLs are unique.
- Non-empty reconstructed discussions: 127.
- High-confidence secret findings: 0.
- Identity, membership, and label projections are preserved.
