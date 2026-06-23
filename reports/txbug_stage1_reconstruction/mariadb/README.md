# TXBug MariaDB JIRA Stage 1 reconstruction

This directory contains a reconstructed Stage 1 candidate set for the MariaDB JIRA portion of TXBug.

## Method

- Source: `https://jira.mariadb.org` REST API.
- Projects: `MDEV, MCOL`.
- Created window: `2018-01-01` to `2022-12-31`.
- JIRA text keywords: `transaction`, `transactions`, `rollback`, `roll back`, `isolation level`, `serializable`, `read committed`, `repeatable read`, `XA`, `deadlock`, `commit transaction`, `abort transaction`.
- Candidate rows are deduplicated by JIRA issue URL; `matched_keywords` records all keywords that matched each issue.
- `body` is the JIRA description. `comments` concatenates public JIRA comments with timestamp/author markers.

## Output

- `txbug_mariadb_candidates.csv`: 4113 unique candidate rows.
- `txbug_mariadb_candidates.raw.jsonl`: raw REST payloads plus matched keywords.
- `fetch_manifest.json`: query totals and fetch parameters.
- `mariadb_final_coverage_audit.csv`: coverage of the local final TXBug MariaDB rows.

## Counts

- mariadb: 3918
- mariadb_columnstore: 195

## Coverage against local final TXBug MariaDB rows

- Covered: 24/24
- Missing: 0
