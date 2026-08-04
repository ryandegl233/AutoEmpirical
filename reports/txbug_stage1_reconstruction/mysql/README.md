# TXBug MySQL Bugs Stage 1 Reconstruction

_Generated on 2026-06-23; retained as source-specific provenance after the full
7,775-record integration._

## Access method

Direct command-line requests to `bugs.mysql.com` returned Oracle/Akamai 403
responses. Public search and detail pages were therefore read through the
in-app browser using GET requests only.

## Search strategy

Searches used all statuses, included comments, ordered by ID, and locally
filtered dates to 2018-01-01 through 2022-12-31. Keywords covered transaction,
rollback, isolation levels, XA, deadlock, InnoDB transaction tables, and
commit/abort variants.

## Result

| Item | Count |
| --- | ---: |
| Candidate rows | 773 |
| Rows with detail text | 773 / 773 |
| Rows with extracted comments | 772 / 773 |
| Final MySQL cohort | 33 |
| Final cohort covered by search | 31 / 33 |

The two final rows not returned by the search are MySQL Bugs `104833` and
`92993`. They remain in the integrated data with
`comments_unavailable_in_source`, not a verified zero-comment status.

## Files

The directory retains normalized candidates, text-enhanced candidates, parsed
search rows, detail JSON, per-record audit, and the fetch manifest.

## Completion status

All remaining sources were subsequently reconciled and the complete Stage 1
discussion repair was integrated. See the canonical
[Stage 1 report](../../txbug_stage1_information_reconstruction/README.md) and
[Stage 2/3 report](../../txbug_information_reconstruction/README.md).
