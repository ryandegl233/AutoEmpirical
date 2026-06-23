# TXBug PostgreSQL Stage 1 reconstruction

This directory contains a reconstructed Stage 1 candidate set for PostgreSQL bug-report emails.

## Method

- Source: `https://www.postgresql.org/list/pgsql-bugs` monthly `pgsql-bugs` archives.
- Archive window: `2018-01` through `2022-12`.
- Initial candidate universe: original `BUG #...` reports, excluding `Re:` replies.
- Keyword filter over title + message body: `transaction`, `transactions`, `rollback`, `roll back`, `isolation level`, `serializable`, `read committed`, `repeatable read`, `XA`, `deadlock`, `commit transaction`, `abort transaction`.
- `body` is the original bug-report email content. `comments` concatenates response emails linked by the archive page.

## Output

- `txbug_postgresql_candidates.csv`: 437 unique candidate rows.
- `txbug_postgresql_candidates.raw.jsonl`: raw parsed message payloads.
- `fetch_manifest.json`: fetch parameters and counts.
- `postgresql_final_coverage_audit.csv`: coverage of local final TXBug PostgreSQL rows.

## Coverage against local final TXBug PostgreSQL rows

- Covered: 0/6
- Missing: 6
  - https://www.postgresql.org/message-id/15875-76bf5472863f6ce3@postgresql.org
  - https://www.postgresql.org/message-id/15946-5c7570a2884a26cf@postgresql.org
  - https://www.postgresql.org/message-id/16676-fd62c3c835880da6@postgresql.org
  - https://www.postgresql.org/message-id/16771-cbef7d97ba93f4b9@postgresql.org
  - https://www.postgresql.org/message-id/17116-d6ca217acc180e30@postgresql.org
  - https://www.postgresql.org/message-id/17385-9ee529fb091f0ce5@postgresql.org
