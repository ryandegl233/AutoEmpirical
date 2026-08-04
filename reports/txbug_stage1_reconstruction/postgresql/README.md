# TXBug PostgreSQL Stage 1 Reconstruction

_Retained as source-specific provenance after the full 7,775-record
integration._

## Method

- Source: monthly `pgsql-bugs` archives from 2018-01 through 2022-12.
- Candidate universe: original `BUG #...` reports, excluding `Re:` replies.
- Keyword filter: transaction, rollback, isolation levels, XA, deadlock, and
  transaction commit/abort variants.
- `body` is the original report; `comments` concatenates linked replies.

## Outputs

- `txbug_postgresql_candidates.csv`: 437 unique candidates.
- `txbug_postgresql_candidates.raw.jsonl`: parsed message payloads.
- `fetch_manifest.json`: parameters and counts.
- `postgresql_final_coverage_audit.csv`: final-cohort coverage.

## Initial coverage

The initial keyword reconstruction covered 0/6 final PostgreSQL rows. Those
six mailing-list message IDs are retained in the coverage audit.

## Completion status

The six final mailing-list threads were incorporated during the later full
discussion reconstruction; the integrated final cohort reports discussion for
all six. See the canonical
[Stage 1 report](../../txbug_stage1_information_reconstruction/README.md) and
[Stage 2/3 report](../../txbug_information_reconstruction/README.md).
