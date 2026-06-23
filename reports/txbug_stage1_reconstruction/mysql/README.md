# TXBug MySQL Bugs Stage 1 Reconstruction

Generated on 2026-06-23.

This directory contains the MySQL Bugs part of the Stage 1 reconstruction for:

```text
icse2024_understanding_transaction_bugs_in_database
```

The source site is:

```text
https://bugs.mysql.com/
```

## Access Note

Direct command-line HTTP requests to `bugs.mysql.com` returned Oracle/Akamai 403 responses from this environment. The candidate set was therefore collected through the in-app browser by reading public search-result and bug-detail pages. No forms with side effects were submitted; only GET pages were loaded.

## Search Strategy

The reconstruction uses MySQL Bugs search pages with:

- `status[]=All`
- `include_comments=on`
- `order_by=id`
- `direction=DESC`
- issue date filtered locally to 2018-01-01 through 2022-12-31

Core terms:

- `transaction`
- `transactions`
- `rollback`
- `"roll back"`
- `"isolation level"`
- `serializable`
- `"read committed"`
- `"repeatable read"`
- `XA`
- `deadlock`
- `innodb_trx`
- `data_locks`
- `"commit transaction"`
- `"transaction commit"`
- `"abort transaction"`
- `"transaction abort"`

Supplement terms tried:

- `"read uncommitted"`
- `"xa rollback"`
- `"XA rollback"`

## Result

| Item | Count |
| --- | ---: |
| MySQL candidate rows | 773 |
| Candidate rows with detail text | 773 / 773 |
| Candidate rows with extracted comments | 772 / 773 |
| Local final MySQL TXBug rows | 33 |
| Covered local final rows | 31 / 33 |

The detail crawl extracted public bug-detail text into `body` and `comments` fields. The enhanced file has non-empty `body` for every candidate row, with average body length around 3.2k characters and average comments length around 4.0k characters.

The two local final MySQL TXBug rows not returned by the current search reconstruction are:

```text
https://bugs.mysql.com/bug.php?id=104833
https://bugs.mysql.com/bug.php?id=92993
```

Both are present in the existing Stage 2/Stage 3 data and should be retained later as Stage 2 lineage rows when integrating the reconstructed TXBug Stage 1 candidate set.

## Files

| File | Meaning |
| --- | --- |
| `txbug_mysql_candidates.csv` | Normalized MySQL Bugs candidate table |
| `txbug_mysql_candidates_with_text.csv` | Enhanced candidate table with extracted `body`, `comments`, `description`, `how_to_repeat`, and `suggested_fix` fields |
| `txbug_mysql_candidates.search_rows.jsonl` | Raw search-result rows parsed from MySQL Bugs result pages |
| `details/txbug_mysql_candidate_details.json` | Robust JSON detail dump for all 773 candidates |
| `details/mysql_detail_fetch_audit.csv` | Per-candidate detail-fetch status and text-length audit |
| `fetch_manifest.json` | Keywords, counts, and reconstruction metadata |

## Next Step

Continue with the remaining non-GitHub TXBug sources:

1. MariaDB JIRA
2. PostgreSQL
3. SQLite

After all source-specific candidate sets are available, merge them with the GitHub and MySQL candidates, compare the total against the paper's 7,775 Stage 1 count, and then replace the current TXBug placeholder rows in `Dataset/stage1.csv`.
