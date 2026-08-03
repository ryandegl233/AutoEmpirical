# TXBug MariaDB JIRA Stage 1 Reconstruction

_Retained as source-specific provenance after the full 7,775-record
integration._

## 🔎 Method

- Source: `https://jira.mariadb.org` REST API.
- Projects: `MDEV` and `MCOL`.
- Created window: 2018-01-01 through 2022-12-31.
- Keywords cover transactions, rollback, isolation, XA, deadlock, and
  transaction commit/abort variants.
- Rows are deduplicated by JIRA issue URL.
- `body` is the JIRA description; `comments` concatenates public comments with
  timestamp and author markers.

## 📊 Outputs and counts

| Artifact or group | Count |
| --- | ---: |
| `txbug_mariadb_candidates.csv` | 4,113 |
| MariaDB candidates | 3,918 |
| MariaDB ColumnStore candidates | 195 |
| Final cohort coverage | 24 / 24 |

The directory also retains raw REST payloads, a fetch manifest, and
`mariadb_final_coverage_audit.csv`.

## ✅ Completion status

The full multi-source Stage 1 discussion repair was subsequently integrated.
See the canonical
[integration report](../../txbug_stage1_information_reconstruction/README.md).
