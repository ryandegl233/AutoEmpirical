# ICSE 2024 Transaction Bugs discussion reconstruction

This reconstruction restores public discussion evidence for the 140-record
Stage 2/3 cohort while preserving record identity, stage membership, and all
gold labels.

## Version semantics

The author artifact does not contain a frozen copy of every source discussion.
Recovered source text is therefore classified as `current_unversioned`. It is
currently visible public evidence, not a claim that every comment existed at
the paper's original collection cutoff.

## Source coverage

- `github_issue`: 71 records, 68 with discussion
- `mariadb_jira`: 24 records, 19 with discussion
- `mysql_bugs`: 33 records, 31 with discussion
- `postgresql_mail_thread`: 6 records, 6 with discussion
- `sqlite_fossil_ticket`: 6 records, 3 with discussion

## Retrieval status

- `ok`: 127
- `ok_zero_comments`: 11
- `source_unavailable`: 2

`no_comments_in_source` means the authoritative source or retained source
reconstruction exposes zero comments. `comments_unavailable_in_source` means
the discussion could not be recovered and must not be interpreted as zero.

## Integrity

- Evidence records: 140
- Unique record IDs: true
- Unique issue URLs: true
- Non-empty reconstructed discussions: 127
- High-confidence secret findings: 0
- Immutable identity and label projections preserved for every rewritten file
