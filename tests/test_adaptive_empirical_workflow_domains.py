from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    EvidenceRequest,
    RetrievalStatus,
    SpecialistType,
)
from Benchmark.src.adaptive_empirical_workflow.domains import (
    build_record_runtime,
    load_domain_inputs,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cohort = tmp_path / "cohort.csv"
    with cohort.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "issue_url",
                "title",
                "body",
                "comments",
                "state",
                "created_at",
                "changed_files",
                "code_diff",
                "decision",
                "symptom",
                "root_cause",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "record-1",
                "issue_url": "https://example.test/issues/1",
                "title": "Request rejected",
                "body": "A valid request is rejected before the patch.",
                "comments": "Maintainer confirms this is unintended.",
                "state": "closed",
                "created_at": "2026-01-01T00:00:00Z",
                "changed_files": '["handler.go", "handler_test.go"]',
                "code_diff": (
                    "diff --git a/handler.go b/handler.go\n"
                    "+if entry == nil { continue }\n"
                    "diff --git a/handler_test.go b/handler_test.go\n"
                    "+func TestNullEntry(t *testing.T) {}"
                ),
                "decision": "SECRET_GOLD_DECISION",
                "symptom": "SECRET_GOLD_SYMPTOM",
                "root_cause": "SECRET_GOLD_CAUSE",
            }
        )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "symptom": ["unexpected_rejection"],
                "root_cause": ["missing_null_check"],
            }
        ),
        encoding="utf-8",
    )
    return cohort, taxonomy


def _request(
    specialist: SpecialistType,
    *,
    query: str = "handler null entry",
    target_source: str = "record-local source material",
    max_items: int = 3,
) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=f"request-{specialist.value}",
        missing_fact="Whether the hidden source confirms the exact failure mechanism",
        why_needed="This fact distinguishes the leading hypothesis from its alternative.",
        target_specialist=specialist,
        target_source=target_source,
        query=query,
        expected_decision_impact=(
            "Direct confirmation supports the leading hypothesis; absence or "
            "contradiction supports the nearest alternative classification."
        ),
        max_items=max_items,
    )


def test_domain_loader_adds_stage2_taxonomy_without_changing_gold_rows(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)

    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )

    assert inputs.taxonomy["decision"] == [
        "accepted_fault",
        "rejected_candidate",
    ]
    assert inputs.records[0]["decision"] == "SECRET_GOLD_DECISION"


def test_initial_ledger_includes_small_issue_context_but_withholds_code(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )

    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )
    initial_content = "\n".join(item.content for item in runtime.ledger.view().items)

    assert "Request rejected" in initial_content
    assert "SECRET_GOLD" not in initial_content
    assert "Maintainer confirms" in initial_content
    assert "diff --git" not in initial_content
    assert runtime.ledger.view().items[0].metadata["captured_frozen_source_types"] == (
        "changed_files",
        "code_diff",
        "issue_body",
        "issue_comments",
        "taxonomy",
    )
    assert runtime.ledger.view().items[0].metadata[
        "unavailable_frozen_source_types"
    ] == ("commit_history",)


def test_eager_issue_context_is_not_retrieved_again_but_code_remains_on_demand(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    issue_delta = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="maintainer unintended",
        ),
        runtime.ledger,
    )
    code_delta = runtime.specialists.run(
        _request(SpecialistType.CODE_CONTEXT), runtime.ledger
    )

    assert issue_delta.status is RetrievalStatus.ABSENT
    assert issue_delta.diagnostics["reason"] == "evidence_already_in_ledger"
    assert code_delta.status is RetrievalStatus.FOUND
    assert "handler_test.go" in code_delta.items[0].content


def test_specialist_does_not_return_an_item_already_in_the_ledger(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )
    request = _request(
        SpecialistType.ISSUE_PR,
        query="maintainer unintended",
    )
    first = runtime.specialists.run(request, runtime.ledger)
    for item in first.items:
        runtime.ledger.append(item)

    second = runtime.specialists.run(request, runtime.ledger)

    assert second.status is RetrievalStatus.ABSENT
    assert second.diagnostics["reason"] == "evidence_already_in_ledger"


def test_specialist_returns_only_query_matching_bounded_passages(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    record = {
        **inputs.records[0],
        "comments": (
            "Allocator ownership leaks after each rejected request.\n\n"
            "An unrelated paragraph discusses documentation.\n\n"
            "The allocator is released by the patch."
        ),
    }
    runtime = build_record_runtime(
        record,
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="allocator leak",
            target_source="linked issue comments",
            max_items=1,
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.FOUND
    assert len(delta.items) == 1
    assert "allocator" in delta.items[0].content.casefold()
    assert "unrelated paragraph" not in delta.items[0].content
    assert delta.items[0].metadata["request_id"] == delta.request_id
    assert delta.items[0].metadata["frozen_source_type"] == "issue_comments"


def test_available_source_with_zero_query_overlap_is_absent(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.CODE_CONTEXT,
            query="allocator ownership leak",
            target_source="code diff",
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.ABSENT
    assert delta.diagnostics["reason"] == "no_matching_passage"


def test_missing_frozen_history_is_unavailable_not_absent(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.COMMIT_HISTORY,
            query="allocator regression history",
            target_source="commit history",
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.UNAVAILABLE
    assert delta.diagnostics["reason"] == "source_not_captured"


def test_empty_frozen_history_cell_is_unavailable_not_absent(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    runtime = build_record_runtime(
        {**inputs.records[0], "commit_history": ""},
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.COMMIT_HISTORY,
            query="allocator regression history",
            target_source="commit history",
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.UNAVAILABLE
    assert delta.diagnostics["reason"] == "source_not_captured"


@pytest.mark.parametrize(
    ("comments", "expected_status"),
    [
        (None, RetrievalStatus.UNAVAILABLE),
        ("", RetrievalStatus.UNAVAILABLE),
        ("not_available_in_source", RetrievalStatus.UNAVAILABLE),
        ("  Not_Fetched  ", RetrievalStatus.UNAVAILABLE),
        ("COMMENTS_UNAVAILABLE_IN_SOURCE", RetrievalStatus.UNAVAILABLE),
        ("no_comments_in_source", RetrievalStatus.ABSENT),
        ("[]", RetrievalStatus.ABSENT),
    ],
)
def test_comment_capture_states_do_not_fabricate_sentinel_evidence(
    tmp_path: Path,
    comments: str | None,
    expected_status: RetrievalStatus,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    record = dict(inputs.records[0])
    if comments is None:
        record.pop("comments")
    else:
        record["comments"] = comments
    runtime = build_record_runtime(
        record,
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="allocator ownership leak",
            target_source="issue comments",
        ),
        runtime.ledger,
    )
    initial_content = "\n".join(item.content for item in runtime.ledger.view().items)

    assert delta.status is expected_status
    assert "no_comments_in_source" not in initial_content
    assert "not_available_in_source" not in initial_content
    assert "not_fetched" not in initial_content.casefold()
    assert "comments_unavailable_in_source" not in initial_content.casefold()
    initial_metadata = runtime.ledger.view().items[0].metadata
    assert ("issue_comments" in initial_metadata["captured_frozen_source_types"]) is (
        expected_status is RetrievalStatus.ABSENT
    )
    assert ("issue_comments" in initial_metadata["unavailable_frozen_source_types"]) is (
        expected_status is RetrievalStatus.UNAVAILABLE
    )
    assert all(item.source_type != "issue_comments" for item in runtime.ledger.view().items)


@pytest.mark.parametrize("marker", ["not_fetched", "not_available_in_source", "comments_unavailable_in_source"])
@pytest.mark.parametrize("field_name", ["body", "title", "state", "created_at", "source_project", "issue_url"])
def test_unavailable_record_fields_cannot_become_summary_evidence(field_name, marker):
    record = {"record_id": "r", "title": "Allocator fails on a valid request.",
              "issue_url": "https://example.test/issues/1", field_name: f"  {marker.upper()}  "}
    runtime = build_record_runtime(record, taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
                                   domain="icse2023")
    for item in runtime.ledger.view().items:
        assert marker not in item.content.casefold()
        assert marker not in item.source_uri.casefold()
    assert record[field_name] == f"  {marker.upper()}  "


@pytest.mark.parametrize("marker", ["not_fetched", "not_available_in_source", "comments_unavailable_in_source"])
@pytest.mark.parametrize("field_name,specialist,target,source_type", [
    ("body", SpecialistType.ISSUE_PR, "body", "issue_body"),
    ("comments", SpecialistType.ISSUE_PR, "comments", "issue_comments"),
    ("changed_files", SpecialistType.CODE_CONTEXT, "changed files", "changed_files"),
    ("code_diff", SpecialistType.CODE_CONTEXT, "patch", "code_diff"),
    ("commit_history", SpecialistType.COMMIT_HISTORY, "commit history", "commit_history"),
])
def test_unavailable_source_cannot_be_retrieved_or_marked_captured(marker, field_name, specialist, target, source_type):
    record = {"record_id": "r", "title": "Allocator fails on a valid request.", field_name: f" {marker.upper()} "}
    runtime = build_record_runtime(record, taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
                                   domain="icse2023")
    delta = runtime.specialists.run(_request(specialist, target_source=target, query=marker), runtime.ledger)
    assert delta.status is RetrievalStatus.UNAVAILABLE
    assert not delta.items
    metadata = runtime.ledger.view().items[0].metadata
    assert source_type not in metadata["captured_frozen_source_types"]
    assert source_type in metadata["unavailable_frozen_source_types"]


def test_new_domain_default_inputs_follow_shared_registry(monkeypatch):
    from Benchmark.src import paper_benchmark
    from Benchmark.src.adaptive_empirical_workflow.domains import domain_profile
    monkeypatch.setattr(paper_benchmark, "DEFAULT_OUTPUT_ROOT", Path("Benchmark/results/test-versioned-inputs"))
    for domain in ("fse2021", "icse2021", "icse2022", "icse2023", "icse2024"):
        registered = paper_benchmark.get_paper_profile(domain)
        actual = domain_profile(domain)
        assert actual.default_cohort_path == registered.default_cohort_path
        assert actual.default_taxonomy_path == registered.default_taxonomy_path


def test_marker_words_inside_real_source_text_are_preserved():
    content = "The fetch request fails with status not_fetched despite a valid endpoint."
    runtime = build_record_runtime(
        {"record_id": "r", "title": "Fetch failure", "body": content, "comments": content},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]}, domain="icse2023",
    )
    items = runtime.ledger.view().items
    assert content in items[0].content
    assert next(item for item in items if item.source_type == "issue_comments").content == content
    assert "issue_body" in items[0].metadata["captured_frozen_source_types"]


def test_real_serialized_comments_return_only_matching_comment(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "ase2022", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    record = {
        **inputs.records[0],
        "comments": (
            "['Allocator ownership leak is confirmed.====='; "
            "'An unrelated rendering comment follows.=====']"
        ),
    }
    runtime = build_record_runtime(
        record,
        taxonomy=inputs.taxonomy,
        domain="ase2022",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="allocator ownership leak",
            target_source="issue comments",
            max_items=1,
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.FOUND
    assert len(delta.items) == 1
    assert "Allocator ownership leak" in delta.items[0].content
    assert "unrelated rendering" not in delta.items[0].content


def test_issue_body_and_comments_are_routed_as_distinct_frozen_sources(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    record = {
        **inputs.records[0],
        "body": "Allocator ownership leak appears in the issue body.",
        "comments": "Maintainer discusses unrelated rendering behavior.",
    }
    runtime = build_record_runtime(
        record,
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    body_delta = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="allocator ownership leak",
            target_source="issue body",
            max_items=1,
        ),
        runtime.ledger,
    )
    wrong_subsource = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="allocator ownership leak",
            target_source="issue comments",
            max_items=1,
        ),
        runtime.ledger,
    )

    assert body_delta.status is RetrievalStatus.FOUND
    assert body_delta.items[0].source_type == "issue_body"
    assert wrong_subsource.status is RetrievalStatus.ABSENT


def test_same_issue_passage_is_deduplicated_within_one_delta(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    runtime = build_record_runtime(
        {
            **inputs.records[0],
            "body": "Allocator ownership leak.",
            "comments": '["Allocator ownership leak."]',
        },
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.ISSUE_PR,
            query="allocator ownership leak",
            target_source="linked issue",
            max_items=3,
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.FOUND
    assert len(delta.items) == 1


def test_code_and_test_specialists_route_to_matching_diff_passages(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    code_delta = runtime.specialists.run(
        _request(
            SpecialistType.CODE_CONTEXT,
            query="nil continue",
            target_source="code diff",
            max_items=1,
        ),
        runtime.ledger,
    )
    test_delta = runtime.specialists.run(
        _request(
            SpecialistType.TEST_EVIDENCE,
            query="null entry test",
            target_source="test diff",
            max_items=1,
        ),
        runtime.ledger,
    )

    assert code_delta.status is RetrievalStatus.FOUND
    assert "entry == nil" in code_delta.items[0].content
    assert "TestNullEntry" not in code_delta.items[0].content
    assert test_delta.status is RetrievalStatus.FOUND
    assert "TestNullEntry" in test_delta.items[0].content
    assert "entry == nil" not in test_delta.items[0].content


def test_code_diff_identity_is_stable_and_deduplicated_across_specialists(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )
    query = "null entry test"
    code_delta = runtime.specialists.run(
        _request(
            SpecialistType.CODE_CONTEXT,
            query=query,
            target_source="code diff",
            max_items=1,
        ),
        runtime.ledger,
    )
    for item in code_delta.items:
        runtime.ledger.append(item)

    replay = runtime.specialists.run(
        _request(
            SpecialistType.TEST_EVIDENCE,
            query=query,
            target_source="test diff",
            max_items=1,
        ),
        runtime.ledger,
    )
    fresh_runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )
    fresh_test = fresh_runtime.specialists.run(
        _request(
            SpecialistType.TEST_EVIDENCE,
            query=query,
            target_source="test diff",
            max_items=1,
        ),
        fresh_runtime.ledger,
    )

    assert code_delta.status is RetrievalStatus.FOUND
    assert code_delta.items[0].source_type == "code_diff"
    assert fresh_test.items[0].source_type == "code_diff"
    assert fresh_test.items[0].evidence_id == code_delta.items[0].evidence_id
    assert replay.status is RetrievalStatus.ABSENT
    assert replay.diagnostics["reason"] == "evidence_already_in_ledger"


def test_target_source_mismatch_remains_a_fatal_contract_error(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    with pytest.raises(ValueError, match="target_source"):
        runtime.specialists.run(
            _request(
                SpecialistType.CODE_CONTEXT,
                query="maintainer unintended",
                target_source="issue comments",
            ),
            runtime.ledger,
        )


def test_taxonomy_specialist_returns_only_matching_codebook_entry(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024",
        cohort_path=cohort,
        taxonomy_path=taxonomy_path,
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.TAXONOMY_KNOWLEDGE,
            query="missing null check",
            target_source="supplied taxonomy",
            max_items=1,
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.FOUND
    assert len(delta.items) == 1
    assert "missing_null_check" in delta.items[0].content
    assert "unexpected_rejection" not in delta.items[0].content


def test_taxonomy_specialist_retrieves_domain_boundary_guidance(
    tmp_path: Path,
) -> None:
    cohort, taxonomy_path = _write_inputs(tmp_path)
    inputs = load_domain_inputs(
        "issta2024", cohort_path=cohort, taxonomy_path=taxonomy_path
    )
    runtime = build_record_runtime(
        inputs.records[0],
        taxonomy=inputs.taxonomy,
        domain="issta2024",
        retrieved_at="2026-08-05T12:45:00Z",
    )

    delta = runtime.specialists.run(
        _request(
            SpecialistType.TAXONOMY_KNOWLEDGE,
            query="directly observed runtime outcome",
            target_source="supplied taxonomy",
            max_items=1,
        ),
        runtime.ledger,
    )

    assert delta.status is RetrievalStatus.FOUND
    assert "directly observed runtime outcome" in delta.items[0].content.casefold()
