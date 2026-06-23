# TXBug GitHub Stage 1 Reconstruction

Generated on 2026-06-23.

This directory contains the GitHub-sourced part of the Stage 1 reconstruction for:

```text
icse2024_understanding_transaction_bugs_in_database
```

The TXBug paper studies six DBMSs. This GitHub pass covers only:

| DBMS | Repository |
| --- | --- |
| TiDB | `pingcap/tidb` |
| CockroachDB | `cockroachdb/cockroach` |

The paper's full Stage 1 candidate count is 7,775 across GitHub, MySQL Bugs, MariaDB JIRA, PostgreSQL, and SQLite sources.

## Why the Scope Was Narrowed

A literal keyword replay with broad terms such as `commit` and `abort` is too noisy for GitHub Search API. For example, `commit` alone returns thousands of ordinary development issues that are not transaction-bug candidates. It also makes the API crawl slow because GitHub Search is rate-limited.

The current practical reconstruction therefore uses a narrower transaction-oriented keyword set and a TiDB label supplement.

## Current Recommended GitHub Candidate Set

Use:

```text
core_plus_tidb_label/txbug_github_candidates.csv
```

It combines:

1. Core transaction keywords:
   - `transaction`
   - `transactions`
   - `rollback`
   - `"roll back"`
   - `"isolation level"`
   - `serializable`
   - `"commit transaction"`
   - `"transaction commit"`
   - `"abort transaction"`
   - `"transaction abort"`
   - `XA`
2. TiDB supplement:
   - `repo:pingcap/tidb is:issue created:2018-01-01..2022-12-31 label:"sig/transaction"`

## Result

| Candidate set | Total | TiDB | CockroachDB | Local GitHub TXBug coverage |
| --- | ---: | ---: | ---: | ---: |
| `core` | 4,148 | 759 | 3,389 | 55 / 71 |
| `core_plus_tidb_label` | 4,335 | 946 | 3,389 | 70 / 71 |

The one local GitHub TXBug not covered by `core_plus_tidb_label` is:

```text
https://github.com/pingcap/tidb/issues/39851
```

Current GitHub metadata for that issue uses `sig/planner`, not `sig/transaction`, and it is not returned by the core transaction keyword queries. It should be preserved later as a Stage 2 lineage row when integrating the reconstructed Stage 1.

## Files

| File | Meaning |
| --- | --- |
| `core/txbug_github_candidates.csv` | Core keyword GitHub candidate set |
| `core/txbug_github_candidates.raw.jsonl` | Raw GitHub payloads for the core set |
| `core/fetch_manifest.json` | Query manifest for the core set |
| `core_plus_tidb_label/txbug_github_candidates.csv` | Recommended GitHub candidate set after TiDB label supplement |
| `core_plus_tidb_label/tidb_sig_transaction.raw.jsonl` | Raw payloads for the TiDB label supplement |
| `core_plus_tidb_label/fetch_manifest.json` | Manifest for the merged recommended set |

## Reproduction Commands

```powershell
python scripts\fetch_txbug_github_stage1_candidates.py --keyword-mode core
```

The TiDB label supplement was generated with:

```text
repo:pingcap/tidb is:issue created:2018-01-01..2022-12-31 label:"sig/transaction"
```

## Next Step

Continue reconstructing non-GitHub TXBug sources:

1. MySQL Bugs
2. MariaDB JIRA
3. PostgreSQL
4. SQLite

After those are reconstructed, merge all source-specific candidate sets, compare against the paper's 7,775 Stage 1 count, and integrate them into `Dataset/stage1.csv` with a small lineage supplement for any final Stage 2/3 rows not found in the reconstructed candidates.
