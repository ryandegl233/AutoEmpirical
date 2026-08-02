# FSE 2021 Autopilot evidence reconstruction

This reconstruction restores public GitHub issue and pull-request evidence for the paper cohort while preserving all record identities, stage membership, provenance, and gold labels.

## Version semantics

The source artifact does not publish a historical acquisition cutoff. GitHub content in this reconstruction is therefore classified as `current_unversioned`: it is a reproducible retrieval record of currently visible public content, not a claim that the text exactly matches the authors' original collection date. Each sidecar record contains its retrieval timestamp and resolved GitHub URL.

## Coverage

- Stage counts: 567 / 168 / 142
- Successful GitHub acquisitions: 567 / 567
- Explicit unresolved records: 0
- Explicit source-URL corrections: 1
- Enriched fields: title, body, comments, created_at, updated_at, state
- Discussion coverage: issue comments plus pull-request review summaries and inline review comments that are currently visible through the GitHub API

The machine-readable invariants and any unresolved URLs are recorded in `audit.json`. The committed evidence sidecar intentionally contains no gold taxonomy labels.

## Source cohort audit

The author replication package is archived at Zenodo DOI
`10.5281/zenodo.4898868`. The retrieved file
`bugSetAndTaxonomy-3 2.zip` has MD5
`118b8494d9c7aecb2435c586ca272f65`. Its bug-set sheet contains 569 rows
but 567 unique GitHub URLs: PX4 issues `3264` and `7696` each occur twice.
The dataset's 567-record Stage 1 cohort therefore covers every unique source
URL; the difference from the paper's 569-row count is deduplication, not
missing objects.

## Approved post-reconstruction security remediation

The final integration also redacts one credential-bearing public comment
body inherited from the previously integrated DL dataset. This approved
non-Autopilot change affects one record in unified Stage 1 and Stage 2 and
is documented in
`../dl_performance_reconstruction/audit/credential_redaction.json`. It does
not change Autopilot records, DL labels, record identity, or provenance.
