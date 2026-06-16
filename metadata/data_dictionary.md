# AutoEmpirical Phase-1 Data Dictionary

## `record_id`

Stable SHA1-based identifier scoped by paper_id.

## `paper_id`

Stable identifier derived from venue, year, and paper title in the source manifest.

## `source_project`

Project or ecosystem that produced the bug report.

## `artifact_type`

Primary artifact class, e.g., bug_report, mixed, secondary_artifact.

## `software_domain`

Coarse software domain inferred from the empirical study scope.

## `source_platform`

Source platform such as github_issues, jira, bugzilla, stack_overflow.

## `issue_url`

Canonical URL for the bug report when available.

## `title`

Bug report title.

## `body`

Bug report body or description.

## `comments`

Concatenated comments or discussion text.

## `created_at`

UTC ISO timestamp when available.

## `updated_at`

UTC ISO timestamp when available.

## `state`

Original issue state.

## `original_labels`

JSON object containing labels exactly as provided or derived from the source dataset.

## `normalized_labels`

JSON object with coarse cross-paper labels and mapping metadata.

## `collection_metadata`

JSON object with conversion provenance and source-row details.
