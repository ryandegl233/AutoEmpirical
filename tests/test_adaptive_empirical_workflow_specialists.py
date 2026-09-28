from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from Benchmark.src.adaptive_empirical_workflow import specialists as specialists_module
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    EvidenceDelta,
    EvidenceExplicitness,
    EvidenceFact,
    EvidenceItem,
    EvidenceRequest,
    RetrievalStatus,
    SpecialistType,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.specialists import (
    SpecialistRegistry,
    apply_evidence_delta,
    evidence_request_key,
    extract_query_passages,
    merge_duplicate_requests,
)


def _request(
    request_id: str,
    missing_fact: str = "Whether the linked issue reports incorrect runtime behavior",
) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=request_id,
        missing_fact=missing_fact,
        why_needed=(
            "A confirmed failure distinguishes a fault repair from maintenance"
        ),
        target_specialist=SpecialistType.ISSUE_PR,
        target_source="linked issue and pull request",
        query="record-1 linked issue incorrect behavior",
        expected_decision_impact=(
            "A maintainer-confirmed failure supports accepted_fault; an enhancement "
            "request without failure evidence supports rejected_candidate"
        ),
        max_items=3,
    )


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the supplied issue.",
        taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
        domain_profile="ase2022",
    )


def _retrieved_item() -> EvidenceItem:
    content = "Maintainer: this is a reproducible runtime failure."
    return EvidenceItem(
        evidence_id="discussion-1",
        record_id="record-1",
        source_type="issue_comment",
        source_uri="https://example.test/issues/1#comment-1",
        retrieved_at="2026-08-05T08:10:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def test_specialist_delta_schema_rejects_classification_labels() -> None:
    with pytest.raises(ValidationError, match="decision"):
        EvidenceDelta.model_validate(
            {
                "request_id": "req-1",
                "specialist": "issue_pr",
                "status": "absent",
                "items": [],
                "facts": [],
                "counterfacts": [],
                "diagnostics": {},
                "decision": "accepted_fault",
            }
        )


def test_duplicate_requests_are_merged_without_losing_team_provenance() -> None:
    merged = merge_duplicate_requests(
        [
            _request("A-req"),
            _request(
                "B-req",
                "  whether THE linked issue reports incorrect runtime behavior  ",
            ),
        ]
    )

    assert len(merged) == 1
    assert merged[0].source_request_ids == ("A-req", "B-req")
    assert merged[0].request.request_id == "A-req"


def test_request_identity_preserves_normalized_query_and_item_bound() -> None:
    original = _request("A-req")
    same_semantics = original.model_copy(
        update={
            "request_id": "B-req",
            "query": "incorrect behavior linked ISSUE record 1",
        }
    )
    different_query = original.model_copy(
        update={"request_id": "C-req", "query": "record-1 null guard"}
    )
    different_bound = original.model_copy(
        update={"request_id": "D-req", "max_items": 4}
    )

    assert evidence_request_key(original) == evidence_request_key(same_semantics)
    assert evidence_request_key(original) != evidence_request_key(different_query)
    assert evidence_request_key(original) != evidence_request_key(different_bound)
    assert [
        group.source_request_ids
        for group in merge_duplicate_requests(
            [original, same_semantics, different_query, different_bound]
        )
    ] == [("A-req", "B-req"), ("C-req",), ("D-req",)]


def test_registry_returns_unavailable_when_specialist_is_not_configured() -> None:
    delta = SpecialistRegistry().run(_request("req-1"), _ledger())

    assert delta.status is RetrievalStatus.UNAVAILABLE
    assert delta.items == ()
    assert delta.diagnostics["reason"] == "specialist_not_registered"


def test_registry_rejects_a_specialist_response_for_another_request() -> None:
    def wrong_request_specialist(
        request: EvidenceRequest, ledger_view: object
    ) -> EvidenceDelta:
        return EvidenceDelta(
            request_id="different-request",
            specialist=SpecialistType.ISSUE_PR,
            status=RetrievalStatus.ABSENT,
        )

    registry = SpecialistRegistry({SpecialistType.ISSUE_PR: wrong_request_specialist})

    with pytest.raises(ValueError, match="request_id"):
        registry.run(_request("req-1"), _ledger())


def test_passage_extraction_is_bounded_by_overlap_and_stable_source_order() -> None:
    matches = extract_query_passages(
        (
            "Allocator pressure is visible in the cache.\n\n"
            "An unrelated paragraph describes rendering.\n\n"
            "The leak grows after every request.\n\n"
            "Allocator telemetry confirms the leak."
        ),
        query="allocator leak",
        max_items=3,
        source_kind="prose",
    )

    assert [match.content for match in matches] == [
        "Allocator telemetry confirms the leak.",
        "Allocator pressure is visible in the cache.",
        "The leak grows after every request.",
    ]
    assert [match.score for match in matches] == [2, 1, 1]
    assert [match.passage_index for match in matches] == [3, 0, 2]


def test_passage_extraction_splits_diff_hunks_before_query_scoring() -> None:
    matches = extract_query_passages(
        (
            "diff --git a/allocator.c b/allocator.c\n"
            "--- a/allocator.c\n"
            "+++ b/allocator.c\n"
            "@@ -1,2 +1,3 @@\n"
            "+release_allocator(block);\n"
            "@@ -20,2 +21,3 @@\n"
            "+refresh_rendering();\n"
            "diff --git a/unrelated.c b/unrelated.c\n"
            "--- a/unrelated.c\n"
            "+++ b/unrelated.c\n"
            "@@ -1 +1 @@\n"
            "+leave_unchanged();"
        ),
        query="release allocator",
        max_items=1,
        source_kind="diff",
    )

    assert len(matches) == 1
    assert "release_allocator" in matches[0].content
    assert "refresh_rendering" not in matches[0].content
    assert "unrelated.c" not in matches[0].content


def test_comment_extraction_splits_real_serialized_comment_boundaries() -> None:
    matches = extract_query_passages(
        (
            "['Allocator ownership leak is confirmed.====='; "
            "'An unrelated rendering comment follows.=====']"
        ),
        query="allocator ownership leak",
        max_items=1,
        source_kind="comments",
    )

    assert len(matches) == 1
    assert "Allocator ownership leak" in matches[0].content
    assert "unrelated rendering" not in matches[0].content


def test_registry_contains_ordinary_runtime_failures_deterministically() -> None:
    def broken_specialist(
        request: EvidenceRequest, ledger_view: object
    ) -> EvidenceDelta:
        raise specialists_module.RecoverableSpecialistError(
            "machine-specific detail must not leak"
        )

    delta = SpecialistRegistry({SpecialistType.ISSUE_PR: broken_specialist}).run(
        _request("req-1"), _ledger()
    )

    assert delta.status is RetrievalStatus.UNAVAILABLE
    assert delta.diagnostics == {
        "reason": "specialist_runtime_failure",
        "exception_type": "RecoverableSpecialistError",
    }


@pytest.mark.parametrize(
    "error",
    [
        KeyError("typoed_field"),
        AttributeError("missing attribute"),
        IndexError("missing item"),
        NotImplementedError("unfinished specialist"),
        RuntimeError("unexpected runtime failure"),
    ],
)
def test_registry_keeps_unexpected_programmer_failures_fatal(
    error: Exception,
) -> None:
    def broken_specialist(
        request: EvidenceRequest, ledger_view: object
    ) -> EvidenceDelta:
        raise error

    with pytest.raises(type(error)):
        SpecialistRegistry({SpecialistType.ISSUE_PR: broken_specialist}).run(
            _request("req-1"), _ledger()
        )


def test_registry_keeps_explicit_invariant_failures_fatal() -> None:
    def invalid_specialist(
        request: EvidenceRequest, ledger_view: object
    ) -> EvidenceDelta:
        raise ValueError("frozen source invariant failed")

    with pytest.raises(ValueError, match="frozen source invariant"):
        SpecialistRegistry({SpecialistType.ISSUE_PR: invalid_specialist}).run(
            _request("req-1"), _ledger()
        )


def test_registry_keeps_malformed_specialist_returns_fatal() -> None:
    def malformed_specialist(request: EvidenceRequest, ledger_view: object) -> object:
        return None

    with pytest.raises(TypeError, match="EvidenceDelta"):
        SpecialistRegistry(
            {SpecialistType.ISSUE_PR: malformed_specialist}  # type: ignore[dict-item]
        ).run(_request("req-1"), _ledger())


def test_found_delta_appends_exact_evidence_and_facts_to_ledger() -> None:
    item = _retrieved_item()

    def issue_specialist(
        request: EvidenceRequest, ledger_view: object
    ) -> EvidenceDelta:
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=SpecialistType.ISSUE_PR,
            status=RetrievalStatus.FOUND,
            items=[item],
            facts=[
                EvidenceFact(
                    claim="A maintainer confirms a reproducible runtime failure.",
                    evidence_ids=[item.evidence_id],
                    explicitness=EvidenceExplicitness.DIRECT,
                )
            ],
        )

    ledger = _ledger()
    delta = SpecialistRegistry({SpecialistType.ISSUE_PR: issue_specialist}).run(
        _request("req-1"), ledger
    )
    appended = apply_evidence_delta(ledger, delta)

    assert appended == ("discussion-1",)
    assert ledger.version == 1
    assert ledger.get("discussion-1").content == item.content
