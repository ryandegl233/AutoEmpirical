import pytest
from pydantic import ValidationError

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    BaselineRevisionAssessment,
    baseline_revision_assessment_digest,
    EvidenceDimension,
    RevisionAssessmentAuditAttempt,
    RevisionConsistencyAuditAttempt,
    RevisionCertificateAuditEntry,
    RevisionProposalAuditEntry,
    RevisionConsistencyReport,
    revision_consistency_report_digest,
    Stage3RevisionAudit,
)
from Benchmark.src.adaptive_empirical_workflow.controller import Stage3WorkflowResult
from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceValidityReport
from Benchmark.src.adaptive_empirical_workflow.audit import _json_value


def _assessment(team: str) -> BaselineRevisionAssessment:
    return BaselineRevisionAssessment(
        assessor_team_id=team,
        proposal_digest="a" * 64,
        dimension="root_cause",
        verdict="revise",
        baseline_label="API Misuse",
        proposed_label="Incorrect Code Logic",
        boundary_card_id="root/api-vs-logic",
        contradicted_baseline_condition="The baseline condition is contradicted.",
        satisfied_proposed_conditions=("The proposed condition is satisfied.",),
        supporting_evidence_ids=("e1",),
        counter_evidence_ids=("e1",),
        baseline_source_config_hash="1" * 64,
        baseline_source_predictions_sha256="2" * 64,
        taxonomy_structure_hash="3" * 64,
        evidence_view_hash="4" * 64,
    )


def _non_revision_assessment(team: str, verdict: str) -> BaselineRevisionAssessment:
    return BaselineRevisionAssessment(
        assessor_team_id=team,
        proposal_digest="a" * 64,
        dimension="root_cause",
        verdict=verdict,
        baseline_source_config_hash="1" * 64,
        baseline_source_predictions_sha256="2" * 64,
        taxonomy_structure_hash="3" * 64,
        evidence_view_hash="4" * 64,
    )


def _consistency(checker: str, owner: str) -> RevisionConsistencyReport:
    return RevisionConsistencyReport(
        checker_team_id=checker,
        assessment_owner_team_id=owner,
        proposal_digest="a" * 64,
        dimension="root_cause",
        baseline_label="API Misuse",
        proposed_label="Incorrect Code Logic",
        status="consistent",
        rationale="The opposite assessment is causally and taxonomically consistent.",
        supporting_evidence_ids=("e1",),
        counter_evidence_ids=("e1",),
        assessment_digest=baseline_revision_assessment_digest(_assessment(owner)),
        boundary_card_id="root/api-vs-logic",
        baseline_source_config_hash="1" * 64,
        baseline_source_predictions_sha256="2" * 64,
        taxonomy_structure_hash="3" * 64,
        evidence_view_hash="4" * 64,
    )


def _revision_audit() -> Stage3RevisionAudit:
    return Stage3RevisionAudit(
        proposal_digests=("a" * 64,),
        routed_card_ids=("root/api-vs-logic",),
        proposal_entries=(
            RevisionProposalAuditEntry(
                proposal_digest="a" * 64,
                dimension="root_cause",
                boundary_card_id="root/api-vs-logic",
                baseline_label="API Misuse",
                proposed_label="Incorrect Code Logic",
            ),
        ),
        assessment_attempts=(
            RevisionAssessmentAuditAttempt(
                proposal_digest="a" * 64,
                dimension=EvidenceDimension.ROOT_CAUSE,
                boundary_card_id="root/api-vs-logic",
                assessor_team_id="A",
                returned_assessment_digest=baseline_revision_assessment_digest(
                    _assessment("A")
                ),
                returned_assessment=_assessment("A"),
                verdict="revise",
                operational_failure=False,
            ),
            RevisionAssessmentAuditAttempt(
                proposal_digest="a" * 64,
                dimension=EvidenceDimension.ROOT_CAUSE,
                boundary_card_id="root/api-vs-logic",
                assessor_team_id="B",
                returned_assessment_digest=None,
                verdict=None,
                operational_failure=True,
            ),
        ),
        dual_revise_proposal_digests=(),
        consistency_attempts=(),
        consistency_pass_proposal_digests=(),
        issued_certificate_digests=(),
        applied_certificate_digests=(),
        ambiguous_certified_dimensions=(),
        failed_proposal_digests=("a" * 64,),
    )


def test_revision_audit_is_deep_frozen_and_json_round_trips() -> None:
    audit = _revision_audit()

    restored = Stage3RevisionAudit.model_validate_json(audit.model_dump_json())

    assert restored == audit
    with pytest.raises(ValidationError):
        restored.assessment_attempts[0].verdict = "preserve"


def test_preservation_result_snapshot_exposes_even_an_empty_revision_audit() -> None:
    empty = Stage3RevisionAudit()
    result = Stage3WorkflowResult(
        reports=(),
        final_decision=None,
        verification=None,
        evidence_deltas=(),
        retrieval_source_request_ids=(),
        budget_exhausted=False,
        validity_report=EvidenceValidityReport(
            valid_evidence_ids=(), capabilities_by_evidence_id={}
        ),
        revision_audit=empty,
    )

    serialized = _json_value(result)

    assert serialized["revision_audit"] == empty.model_dump(mode="json")


def test_revision_audit_rejects_attempts_for_an_unlisted_proposal() -> None:
    with pytest.raises(ValidationError, match="listed proposal"):
        Stage3RevisionAudit(
            assessment_attempts=(
                RevisionAssessmentAuditAttempt(
                    proposal_digest="a" * 64,
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    boundary_card_id="root/api-vs-logic",
                    assessor_team_id="A",
                    operational_failure=True,
                ),
            )
        )


def test_revision_audit_requires_one_routed_card_for_every_proposal() -> None:
    with pytest.raises(ValidationError, match="one routed card"):
        Stage3RevisionAudit(
            proposal_digests=("a" * 64,),
            routed_card_ids=(),
        )


def test_cross_attempt_requires_distinct_checker_and_owner() -> None:
    with pytest.raises(ValidationError, match="self-check"):
        RevisionConsistencyAuditAttempt(
            proposal_digest="a" * 64,
            dimension=EvidenceDimension.ROOT_CAUSE,
            boundary_card_id="root/api-vs-logic",
            checker_team_id="A",
            assessment_owner_team_id="A",
            operational_failure=True,
        )


def test_revision_audit_rejects_cross_pass_without_dual_revise() -> None:
    with pytest.raises(ValidationError, match="consistency pass"):
        Stage3RevisionAudit(
            proposal_digests=("a" * 64,),
            routed_card_ids=("root/api-vs-logic",),
            proposal_entries=(
                RevisionProposalAuditEntry(
                    proposal_digest="a" * 64,
                    dimension="root_cause",
                    boundary_card_id="root/api-vs-logic",
                    baseline_label="API Misuse",
                    proposed_label="Incorrect Code Logic",
                ),
            ),
            consistency_pass_proposal_digests=("a" * 64,),
        )


def test_revision_audit_rejects_attempt_card_that_differs_from_proposal_entry() -> None:
    attempt = (
        _revision_audit()
        .assessment_attempts[0]
        .model_copy(update={"boundary_card_id": "root/wrong-card"})
    )
    with pytest.raises(ValidationError, match="audit attempt|proposal entry"):
        Stage3RevisionAudit(
            **{
                **_revision_audit().model_dump(mode="python"),
                "assessment_attempts": (attempt,),
                "failed_proposal_digests": (),
            }
        )


def test_revision_audit_derives_dual_revise_from_two_returned_assessments() -> None:
    with pytest.raises(ValidationError, match="dual revise"):
        Stage3RevisionAudit(
            proposal_digests=("a" * 64,),
            routed_card_ids=("root/api-vs-logic",),
            proposal_entries=_revision_audit().proposal_entries,
            dual_revise_proposal_digests=("a" * 64,),
        )


def test_revision_audit_rejects_failed_and_consistency_pass_overlap() -> None:
    base = _revision_audit()
    cross_attempts = tuple(
        RevisionConsistencyAuditAttempt(
            proposal_digest="a" * 64,
            dimension="root_cause",
            boundary_card_id="root/api-vs-logic",
            checker_team_id=checker,
            assessment_owner_team_id=owner,
            returned_consistency_digest=revision_consistency_report_digest(
                _consistency(checker, owner)
            ),
            returned_consistency=_consistency(checker, owner),
            status="consistent",
            operational_failure=False,
        )
        for checker, owner in (("A", "B"), ("B", "A"))
    )
    revise_attempts = tuple(
        RevisionAssessmentAuditAttempt(
            proposal_digest="a" * 64,
            dimension="root_cause",
            boundary_card_id="root/api-vs-logic",
            assessor_team_id=team,
            returned_assessment_digest=baseline_revision_assessment_digest(
                _assessment(team)
            ),
            returned_assessment=_assessment(team),
            verdict="revise",
            operational_failure=False,
        )
        for team in ("A", "B")
    )
    with pytest.raises(ValidationError, match="failed proposal"):
        Stage3RevisionAudit(
            proposal_digests=base.proposal_digests,
            routed_card_ids=base.routed_card_ids,
            proposal_entries=base.proposal_entries,
            assessment_attempts=revise_attempts,
            dual_revise_proposal_digests=("a" * 64,),
            consistency_attempts=cross_attempts,
            consistency_pass_proposal_digests=("a" * 64,),
            failed_proposal_digests=("a" * 64,),
        )


def test_revision_audit_rejects_issued_certificate_without_cross_pass() -> None:
    with pytest.raises(ValidationError, match="consistency pass"):
        Stage3RevisionAudit(
            proposal_digests=("a" * 64,),
            routed_card_ids=("root/api-vs-logic",),
            proposal_entries=_revision_audit().proposal_entries,
            issued_certificate_digests=("f" * 64,),
            issued_certificate_entries=(
                RevisionCertificateAuditEntry(
                    certificate_digest="f" * 64,
                    proposal_digest="a" * 64,
                    dimension="root_cause",
                    boundary_card_id="root/api-vs-logic",
                    baseline_label="API Misuse",
                    proposed_label="Incorrect Code Logic",
                ),
            ),
        )


def test_revision_audit_rejects_ambiguity_without_multiple_consistency_passes() -> None:
    with pytest.raises(ValidationError, match="ambiguous"):
        Stage3RevisionAudit(
            ambiguous_certified_dimensions=("root_cause",),
        )


@pytest.mark.parametrize("verdict", ["preserve", "insufficient"])
def test_revision_audit_accepts_non_revision_assessment_without_revision_fields(
    verdict: str,
) -> None:
    returned = _non_revision_assessment("A", verdict)
    audit = Stage3RevisionAudit(
        proposal_digests=("a" * 64,),
        routed_card_ids=("root/api-vs-logic",),
        proposal_entries=_revision_audit().proposal_entries,
        assessment_attempts=(
            RevisionAssessmentAuditAttempt(
                proposal_digest="a" * 64,
                dimension="root_cause",
                boundary_card_id="root/api-vs-logic",
                assessor_team_id="A",
                returned_assessment_digest=baseline_revision_assessment_digest(
                    returned
                ),
                returned_assessment=returned,
                verdict=verdict,
                operational_failure=False,
            ),
            RevisionAssessmentAuditAttempt(
                proposal_digest="a" * 64,
                dimension="root_cause",
                boundary_card_id="root/api-vs-logic",
                assessor_team_id="B",
                operational_failure=True,
            ),
        ),
        failed_proposal_digests=("a" * 64,),
    )

    assert Stage3RevisionAudit.model_validate_json(audit.model_dump_json()) == audit
