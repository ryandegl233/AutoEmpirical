# AutoEmpirical Data Dictionary

This dictionary describes the 23-column schema shared by `Dataset/stage1.csv`,
`Dataset/stage2.csv`, `Dataset/stage3.csv`, and the corresponding
`Dataset/by_paper/<paper_id>/stage*.csv` files.

## `record_id`

Stable SHA1-based record identifier scoped by `paper_id` and source record content.
This field is globally unique across each stage file.

## `paper_id`

Stable identifier for the empirical study that contributed the record.

## `source_project`

Project, ecosystem, repository, product, or study-specific source name associated
with the record.

## `issue_url`

Canonical URL or stable source locator for the issue, bug report, pull request,
message, or source record when available.

## `title`

Bug report title, issue title, or concise source-record summary.

## `body`

Primary bug report body, description, reproduction text, or source-record content.

## `comments`

Concatenated comments, discussion text, extracted source detail, or a source
placeholder such as `no_comments_in_source` when comments were unavailable.

## `created_at`

Creation, submission, or report timestamp when available. Timestamps are preserved
from the best available source and may use source-specific precision or formatting.
Missing timestamps use explicit placeholders such as `not_available_in_source`.

## `updated_at`

Update, modified, close, or last-observed timestamp when available. Values that
cannot be supported by a source timestamp are encoded as `not_available_in_source`
rather than as numeric duration or row-index fields.

## `state`

Original source state or status, such as open, closed, verified, fixed, or a
study-specific status value.

## `symptom`

Observed failure behavior, symptom, consequence, or equivalent source/study label.
Stage 1 and Stage 2 may be partially labeled; Stage 3 is fully labeled.

## `root_cause`

Root-cause label, cause category, taxonomy category, configuration option, or
equivalent source/study label.

## `bug_type`

Source or study-specific bug type/category.

## `component`

Affected framework, project, subsystem, component, or domain category.

## `sub_component`

Source sub-component, subtype, module, or narrower component label.

## `trigger_condition`

Triggering condition, reproduction condition, workload, environment, or relevant
setup required to expose the bug.

## `consequence`

Failure consequence, effect, impact, or downstream behavior.

## `fix_type`

Repair strategy, fix pattern, patch category, or study-specific fixing label.

## `severity_or_impact`

Severity, priority, impact, or equivalent source/study importance label.

## `original_label_json`

JSON object preserving source labels, mapped fields, source-specific metadata, and
conversion provenance needed to trace the normalized row back to the original data.

## `source_file`

Raw source file, export, API source, or reconstruction artifact used by the
converter.

## `source_sheet`

Sheet name, logical table, source partition, or extraction group within
`source_file`.

## `source_row_index`

Zero-based source row index after parser header handling, or the closest available
source-row locator for reconstructed records.
