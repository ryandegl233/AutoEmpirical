# ICSE 2021 IoT discussion reconstruction

This reconstruction restores developer discussion evidence for all 5,548
records in `icse2021_iot_bugs_and_development_challenges` while preserving
record identity, stage membership, author-artifact text, and all gold labels.

## Research objects and source boundary

The dataset contains two GitHub object types:

- 4,697 issues
- 851 pull requests

The official replication package states that its collected-bug artifact stores
URL, title, body, and date, but it does not store comments. Historical artifact
fields remain unchanged. Discussion text was retrieved from the current public
GitHub API and is therefore classified as `current_unversioned`, not as a
frozen January-February 2020 snapshot.

For issues, discussion contains issue comments. For pull requests, it combines
conversation comments, review bodies, and review comments in timestamp order.

## Artifact-to-dataset reconciliation

The file named `5565-collected_bugs.json` currently contains 5,566 rows and
5,545 unique URLs, including 21 duplicate rows. The normalized Stage 1 dataset
contains 5,548 unique URLs: one unique artifact URL is excluded and four URLs
from the analyzed cohort are present in the dataset but absent from that JSON.
The reconstruction is anchored to the normalized 5,548-record dataset rather
than reintroducing duplicate artifact rows.

## Retrieval coverage

- Evidence records: 5,548
- `ok`: 3,728
- `ok_zero_comments`: 1,110
- `source_unavailable`: 710
- issue/pull-request type mismatches: 0

`no_comments_in_source` means the current GitHub source explicitly exposes no
discussion through the applicable channels. `comments_unavailable_in_source`
means GitHub returned a stable 404/410, the repository is unavailable, or issue
tracking is disabled; it must not be interpreted as zero discussion.

## Credential safety

Seven high-confidence credential-shaped values in public issue material were replaced
before publication:

- one private-key block
- one AWS access-key ID
- five Azure IoT Hub shared-access keys

Only typed redaction placeholders are retained. The evidence sidecar and all
rewritten CSVs contain no matching secret material.

## Integrity

- CSV changes are limited to reconstructed `comments` plus three exact
  Azure-key redactions in the `body` fields of two Stage 1 records.
- Net changed rows: 4,438 in Stage 1 and 274 in each of Stage 2 and Stage 3.
- No rows from another paper changed in any unified stage file.
- Per-paper and unified comments agree for every target record.
- Evidence record IDs and issue URLs are unique.
- The evidence sidecar contains no gold-label keys.

Structured evidence is stored in
`Dataset/evidence/icse2021_iot_discussion_evidence.jsonl`. Machine-readable
audit results are in `audit.json`.
