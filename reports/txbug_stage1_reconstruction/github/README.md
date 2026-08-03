# TXBug GitHub Stage 1 Reconstruction

_Generated on 2026-06-23; retained as source-specific provenance after the full
7,775-record integration._

This pass covers GitHub issues from `pingcap/tidb` and
`cockroachdb/cockroach`. The paper's other candidates come from MySQL Bugs,
MariaDB JIRA, PostgreSQL, and SQLite sources.

## 🧭 Query scope

Broad words such as `commit` and `abort` were too noisy for GitHub Search.
The selected set combines transaction-specific keyword queries with TiDB's
`sig/transaction` label.

## 📊 Result

| Candidate set | Total | TiDB | CockroachDB | Final GitHub cohort coverage |
| --- | ---: | ---: | ---: | ---: |
| `core` | 4,148 | 759 | 3,389 | 55 / 71 |
| `core_plus_tidb_label` | 4,335 | 946 | 3,389 | 70 / 71 |

The uncovered final issue is `pingcap/tidb#39851`; current metadata uses
`sig/planner`, so it was later retained through explicit Stage 2 lineage.

## 📁 Files

| Path | Meaning |
| --- | --- |
| `core/` | Core keyword candidate set, raw payloads, and manifest |
| `core_plus_tidb_label/` | Selected merged set, label payloads, and manifest |

## 🔁 Reproduction

```powershell
python scripts\fetch_txbug_github_stage1_candidates.py --keyword-mode core
```

The label supplement used:

```text
repo:pingcap/tidb is:issue created:2018-01-01..2022-12-31 label:"sig/transaction"
```

## ✅ Completion status

All source families were subsequently reconciled and the complete Stage 1
discussion repair was integrated. See the canonical
[Stage 1 report](../../txbug_stage1_information_reconstruction/README.md) and
[Stage 2/3 report](../../txbug_information_reconstruction/README.md).
