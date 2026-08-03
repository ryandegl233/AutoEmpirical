# ICSE 2021 IoT Discussion Reconstruction

_Integrated into `main`; status verified on 2026-08-03._

This reconstruction restores developer discussion for all 5,548 records in
`icse2021_iot_bugs_and_development_challenges` while preserving identity,
membership, author-artifact text, and gold labels.

## 🔎 Research objects and source boundary

The cohort contains 4,697 GitHub issues and 851 pull requests. The official
artifact stores URL, title, body, and date but not comments. Current public
discussion is therefore `current_unversioned`, not a frozen January-February
2020 snapshot.

Issue evidence includes issue comments. Pull-request evidence combines
conversation comments, review bodies, and review comments in timestamp order.

## 🔗 Artifact reconciliation

The file named `5565-collected_bugs.json` contains 5,566 rows and 5,545 unique
URLs, including 21 duplicates. The normalized Stage 1 dataset contains 5,548
unique URLs: one unique artifact URL is excluded and four analyzed-cohort URLs
are present in the dataset but absent from that JSON. Reconstruction is
anchored to the normalized 5,548-record dataset.

## ✅ Retrieval coverage

- Evidence records: 5,548.
- `ok`: 3,728.
- `ok_zero_comments`: 1,110.
- `source_unavailable`: 710.
- Issue/pull-request type mismatches: 0.

`no_comments_in_source` means the current source exposes zero discussion.
`comments_unavailable_in_source` means the source is unavailable or cannot
serve discussion; it is not a verified zero.

The reconstructed discussion is integrated into the applicable unified and
per-paper files. The canonical sidecar is
`../../Dataset/evidence/icse2021_iot_discussion_evidence.jsonl`.

## 🔐 Credential safety

Seven high-confidence credential-shaped values were replaced before
publication: one private-key block, one AWS access-key ID, and five Azure IoT
Hub shared-access keys. Only typed redaction placeholders remain.

## 🧾 Integrity

- CSV changes are limited to reconstructed comments plus three exact Azure-key
  redactions in two Stage 1 body fields.
- Net changed rows: 4,438 in Stage 1 and 274 in each of Stage 2 and Stage 3.
- No rows from another paper changed.
- Per-paper and unified comments agree.
- Evidence IDs and URLs are unique.
- The sidecar contains no gold-label keys.

Machine-readable results are in `audit.json`.
