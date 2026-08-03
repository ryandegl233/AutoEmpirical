# FSE 2021 Autopilot Evidence Reconstruction

_Integrated into `main`; status verified on 2026-08-03._

This reconstruction restores current public GitHub issue and pull-request
evidence while preserving record identities, stage membership, provenance, and
gold labels.

## 🕰️ Version semantics

The source artifact does not publish a historical acquisition cutoff. Retrieved
GitHub content is therefore `current_unversioned`: it is currently visible
public content, not a claim that the text exactly matches the authors'
collection date. Each sidecar record stores its retrieval timestamp and
resolved GitHub URL.

## ✅ Coverage

- Stage counts: 567 / 168 / 142.
- Successful GitHub acquisitions: 567 / 567.
- Explicit unresolved records: 0.
- Explicit source-URL corrections: 1.
- Enriched fields: title, body, comments, created_at, updated_at, and state.
- Discussion channels: issue comments, pull-request review summaries, and
  currently visible inline review comments.

The repaired evidence is integrated into all applicable unified and per-paper
stage files. The canonical sidecar is
`../../Dataset/evidence/fse2021_autopilot_evidence.jsonl`; machine-readable
invariants are in `audit.json`. The sidecar contains no gold taxonomy labels.

## 🔎 Source cohort audit

The author replication package is archived at Zenodo DOI
`10.5281/zenodo.4898868`. The retrieved `bugSetAndTaxonomy-3 2.zip` has MD5
`118b8494d9c7aecb2435c586ca272f65`. Its bug-set sheet contains 569 rows but
567 unique GitHub URLs because PX4 issues `3264` and `7696` each occur twice.
The dataset's 567-record Stage 1 cohort covers every unique source URL; the
difference is deduplication, not missing objects.

## 🔐 Approved security remediation

The final integration also redacts one credential-bearing public comment body
in the previously integrated DL dataset. The approved non-Autopilot change is
documented in
`../dl_performance_reconstruction/audit/credential_redaction.json`. It changes
no Autopilot record and no DL identity, membership, label, or provenance field.
