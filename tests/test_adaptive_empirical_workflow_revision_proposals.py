from __future__ import annotations

import hashlib
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    BaselineAnchor,
    BaselineGateProvenance,
    BaselineRevisionAssessment,
    baseline_revision_assessment_digest,
    BoundaryCard,
    BoundaryChallenge,
    BoundaryChallengeAction,
    BoundaryCriterion,
    CausalConsistencyReport,
    ConsistencyStatus,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceSufficiency,
    EvidenceView,
    DimensionVerificationReport,
    LabelRevisionCertificate,
    RevisionBasis,
    RevisionProposalEnvelope,
    RevisionCandidateSourceKind,
    RevisionAssessmentVerdict,
    RevisionConsistencyReport,
    RootCauseReport,
    Stage3TeamReport,
    Stage3FinalDecision,
    SymptomReport,
    TeamCorrectionAudit,
    TaxonomyNode,
    TaxonomySemanticOrigin,
    TaxonomyStructure,
    VerificationVerdict,
    revision_proposal_digest,
    stage3_team_report_digest,
)
from Benchmark.src.adaptive_empirical_workflow import frozen_evidence_runtime
from Benchmark.src.adaptive_empirical_workflow.domains import build_record_runtime
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph import (
    EvidenceAuthority,
    FrozenEvidenceGraphBundle,
    FrozenGraphNode,
    FrozenNodeType,
    FrozenRecordGraph,
    frozen_graph_node_id,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_runtime import (
    load_registered_frozen_evidence_runtime,
    project_frozen_evidence_for_record,
)
from Benchmark.src.adaptive_empirical_workflow.baseline_anchor import (
    baseline_anchor_hash,
)
from Benchmark.src.adaptive_empirical_workflow.baseline_preservation import (
    apply_baseline_preservation_gate,
    compose_baseline_preservation_decision,
)
from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (
    boundary_challenge_semantic_errors,
    build_revision_proposals,
    compose_baseline_revision_certificates,
    revision_proposal_matches_context,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    taxonomy_structure_hash,
)


def _item(evidence_id: str, source_type: str) -> EvidenceItem:
    content = f"Frozen evidence for {evidence_id}."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="synthetic-record",
        source_type=source_type,
        source_uri=f"https://example.test/{evidence_id}",
        retrieved_at="2026-08-16T00:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def _view() -> EvidenceView:
    return EvidenceView(
        record_id="synthetic-record",
        task="stage3",
        taxonomy={
            "symptom": ["Returned Anomaly", "Execution Halt"],
            "root_cause": ["Caller Contract Violation", "Internal Branch Defect"],
        },
        domain_profile="synthetic-domain",
        ledger_version=7,
        items=(
            _item("symptom-a", "issue_body"),
            _item("symptom-b", "execution_log"),
            _item("cause-a", "code_context"),
            _item("cause-b", "commit_diff"),
            _item("symptom-only", "runtime_observation"),
        ),
    )


def _node(dimension: str, label: str) -> TaxonomyNode:
    reciprocal = {
        "Returned Anomaly": "Execution Halt",
        "Execution Halt": "Returned Anomaly",
        "Caller Contract Violation": "Internal Branch Defect",
        "Internal Branch Defect": "Caller Contract Violation",
    }
    return TaxonomyNode(
        dimension=dimension,
        label=label,
        definition_semantic_origin=TaxonomySemanticOrigin.PAPER_DEFINITION,
        structure_semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
        definition=f"Official definition of {label}.",
        abstraction_level="classification label",
        responsibility_scope="synthetic component",
        concept_kind="outcome" if dimension == "symptom" else "mechanism",
        nearest_neighbors=(reciprocal[label],),
    )


def _card(card_id: str, dimension: str, first: str, second: str) -> BoundaryCard:
    return BoundaryCard(
        card_id=card_id,
        dimension=dimension,
        labels=(first, second),
        semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
        decision_question="Which official definition owns the observed evidence?",
        observable_slots=("mechanism owner",),
        criteria=(
            BoundaryCriterion(
                label=first,
                positive_conditions=(f"evidence satisfies {first}",),
                exclusion_conditions=(f"evidence instead satisfies {second}",),
            ),
            BoundaryCriterion(
                label=second,
                positive_conditions=(f"evidence satisfies {second}",),
                exclusion_conditions=(f"evidence instead satisfies {first}",),
            ),
        ),
    )


def _structure() -> TaxonomyStructure:
    labels = {
        "symptom": ("Returned Anomaly", "Execution Halt"),
        "root_cause": ("Caller Contract Violation", "Internal Branch Defect"),
    }
    return TaxonomyStructure(
        schema_version=1,
        domain="synthetic-domain",
        nodes=tuple(
            _node(dimension, label)
            for dimension, dimension_labels in labels.items()
            for label in dimension_labels
        ),
        boundary_cards=(
            _card("symptom-boundary", "symptom", *labels["symptom"]),
            _card("cause-boundary", "root_cause", *labels["root_cause"]),
        ),
    )


def _anchor(*, valid: bool = True) -> BaselineAnchor:
    return BaselineAnchor(
        record_id="synthetic-record",
        valid=valid,
        symptom_label="Returned Anomaly" if valid else None,
        root_cause_label="Caller Contract Violation" if valid else None,
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def _symptom(label: str, evidence_id: str) -> SymptomReport:
    return SymptomReport(
        label=label,
        behavior_claim="The frozen observation identifies the externally visible outcome.",
        supporting_evidence_ids=(evidence_id,),
        alternative_label=(
            "Execution Halt" if label == "Returned Anomaly" else "Returned Anomaly"
        ),
        boundary_reason="The execution boundary separates returned output from termination.",
        confidence=0.8,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def _root(label: str, evidence_id: str) -> RootCauseReport:
    return RootCauseReport(
        label=label,
        defect_mechanism="The frozen mechanism evidence identifies the responsible component.",
        causal_chain=(
            "A precondition is evaluated.",
            "The responsible component performs the wrong operation.",
            "The observable failure follows.",
        ),
        supporting_evidence_ids=(evidence_id,),
        alternative_label=(
            "Internal Branch Defect"
            if label == "Caller Contract Violation"
            else "Caller Contract Violation"
        ),
        boundary_reason="The ownership boundary distinguishes caller and implementation defects.",
        confidence=0.8,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def _report(
    team_id: str,
    *,
    symptom_label: str = "Returned Anomaly",
    symptom_evidence: str = "symptom-a",
    root_label: str = "Caller Contract Violation",
    root_evidence: str = "cause-a",
) -> Stage3TeamReport:
    return Stage3TeamReport(
        team_id=team_id,
        symptom=_symptom(symptom_label, symptom_evidence),
        root_cause=_root(root_label, root_evidence),
        consistency=CausalConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            rationale="The proposed mechanism is consistent with the observed behavior.",
            supporting_evidence_ids=(symptom_evidence,),
        ),
    )


def _reports_with_b_only_challenger() -> tuple[Stage3TeamReport, Stage3TeamReport]:
    return (
        _report("A"),
        _report("B", root_label="Internal Branch Defect", root_evidence="cause-b"),
    )


def _sealed_candidate_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repair_capabilities: tuple[str, ...] = ("defect_mechanism",),
    repair_authority: EvidenceAuthority = EvidenceAuthority.DIRECT,
    root_labels: tuple[str, ...] = (
        "Caller Contract Violation",
        "Internal Branch Defect",
    ),
    root_card_labels: tuple[str, ...] | None = None,
) -> tuple[
    BaselineAnchor,
    tuple[Stage3TeamReport, Stage3TeamReport],
    TaxonomyStructure,
    EvidenceView,
    str,
    object,
]:
    record_id = "ase2022:owner/repo:91"
    commit = "9" * 40

    def frozen_node(
        node_type: FrozenNodeType,
        *,
        uri: str,
        content: str,
        authority: EvidenceAuthority,
        capabilities: tuple[str, ...],
        immutable_ref: str | None = None,
    ) -> FrozenGraphNode:
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return FrozenGraphNode.model_construct(
            node_id=frozen_graph_node_id(
                record_id=record_id,
                node_type=node_type,
                canonical_uri=uri,
                immutable_ref=immutable_ref,
                content_sha256=content_sha256,
            ),
            record_id=record_id,
            node_type=node_type,
            canonical_uri=uri,
            repository="owner/repo",
            relation_depth=0 if node_type is FrozenNodeType.SEED_ISSUE else 1,
            intrinsic_parent_node_id=None,
            retrieved_at="2026-08-17T00:00:00+00:00",
            remote_updated_at="2026-08-16T00:00:00+00:00",
            immutable_ref=immutable_ref,
            raw_blob_sha256="a" * 64,
            content_sha256=content_sha256,
            content=content,
            evidence_capabilities=capabilities,
            authority=authority,
            metadata=MappingProxyType({}),
        )

    seed = frozen_node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/91",
        content="The API returns a malformed value to the caller.",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )
    repair = frozen_node(
        FrozenNodeType.CHANGED_CODE,
        uri=f"https://github.com/owner/repo/blob/{commit}/src/core.py",
        content="- call_wrong_branch()\n+ call_correct_branch()",
        authority=repair_authority,
        capabilities=repair_capabilities,
        immutable_ref=commit,
    )
    graph = FrozenRecordGraph.model_construct(
        schema_version="ase-frozen-record-graph-v1",
        domain="ase2022",
        record_id=record_id,
        repository="owner/repo",
        seed_node_id=repair.node_id,
        nodes=(repair,),
        edges=(),
        capture_status="captured",
        unavailable_reasons=(),
        graph_sha256="b" * 64,
    )
    manifest = type("Manifest", (), {"bundle_merkle_root": "c" * 64})()
    bundle = FrozenEvidenceGraphBundle.model_construct(
        domain="ase2022",
        bundle_id="candidate-red-v1",
        split_id="candidate-red",
        policy=type("Policy", (), {"policy_id": "one-hop-v2"})(),
        graphs=MappingProxyType({record_id: graph}),
        record_ids=(record_id,),
        manifest=manifest,
        trust_manifest_sha256="d" * 64,
        trust_manifest_relative_path="registered/manifest.json",
    )
    monkeypatch.setattr(
        frozen_evidence_runtime,
        "load_bound_frozen_evidence_graph_for_domain",
        lambda domain, repository_root=None: bundle,
    )
    projection = project_frozen_evidence_for_record(
        load_registered_frozen_evidence_runtime("ase2022"), record_id
    )
    taxonomy = {
        "symptom": ["Returned Anomaly", "Execution Halt"],
        "root_cause": list(root_labels),
    }
    runtime = build_record_runtime(
        {"record_id": record_id, "title": "Synthetic candidate recall"},
        taxonomy=taxonomy,
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    view = runtime.ledger.view()
    symptom_evidence_id = next(
        item.evidence_id for item in view.items if item.evidence_id != repair.node_id
    )
    card_labels = root_card_labels or root_labels[1:]
    base_structure = _structure()
    structure = TaxonomyStructure(
        schema_version=1,
        domain="ase2022",
        nodes=(
            *(node for node in base_structure.nodes if node.dimension == "symptom"),
            *(
                TaxonomyNode(
                    dimension="root_cause",
                    label=label,
                    definition_semantic_origin=TaxonomySemanticOrigin.PAPER_DEFINITION,
                    structure_semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                    definition=f"Official definition of {label}.",
                    abstraction_level="classification label",
                    responsibility_scope="synthetic component",
                    concept_kind="mechanism",
                    nearest_neighbors=(
                        root_labels[1] if label == root_labels[0] else root_labels[0],
                    ),
                )
                for label in root_labels
            ),
        ),
        boundary_cards=(
            base_structure.boundary_cards[0],
            *(
                _card(
                    f"root-boundary-{index}",
                    "root_cause",
                    root_labels[0],
                    label,
                )
                for index, label in enumerate(card_labels, start=1)
            ),
        ),
    )
    anchor = _anchor().model_copy(update={"record_id": record_id})
    reports = tuple(
        _report(team_id).model_copy(
            update={
                "symptom": _symptom("Returned Anomaly", symptom_evidence_id),
                "root_cause": _root(
                    "Caller Contract Violation", repair.node_id
                ).model_copy(update={"boundary_evidence_ids": (repair.node_id,)}),
                "consistency": CausalConsistencyReport(
                    status=ConsistencyStatus.CONSISTENT,
                    rationale="The frozen repair explains the observed returned anomaly.",
                    supporting_evidence_ids=(repair.node_id,),
                ),
            }
        )
        for team_id in ("A", "B")
    )
    return anchor, reports, structure, view, repair.node_id, runtime


def test_sealed_direct_repair_recalls_existing_alternative_without_final_label_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, reports, structure, view, repair_id, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None

    proposals = build_revision_proposals(
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
    )

    assert len(proposals) == 1
    assert proposals[0].proposed_label == "Internal Branch Defect"
    assert proposals[0].candidate_signals[0].source_kinds == (
        RevisionCandidateSourceKind.REPAIR_BACKED_ALTERNATIVE,
    )
    assert proposals[0].candidate_signals[0].authoritative_repair_evidence_ids == (
        repair_id,
    )


def test_boundary_challenger_routes_team_owned_repair_backed_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, reports, structure, view, repair_id, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None
    reports = tuple(
        report.model_copy(
            update={
                "root_cause": report.root_cause.model_copy(
                    update={
                        "boundary_evidence_ids": (),
                        "counter_evidence_ids": (repair_id,),
                    }
                )
            }
        )
        for report in reports
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.CAUSE_REVIEW,
        rationale="The immutable repair should be tested against the nominated boundary.",
        cited_evidence_ids=(repair_id,),
    )

    assert (
        build_revision_proposals(
            anchor=anchor,
            reports=reports,
            structure=structure,
            view=view,
        )
        == ()
    )
    proposals = build_revision_proposals(
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
        boundary_challenge=challenge,
    )

    assert len(proposals) == 1
    assert proposals[0].proposed_label == "Internal Branch Defect"
    assert proposals[0].candidate_signals[0].authoritative_repair_evidence_ids == (
        repair_id,
    )


def test_unknown_pass_challenge_fails_closed_at_proposal_trust_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, reports, structure, view, _, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.PASS,
        rationale="A PASS challenge cannot cite evidence outside the canonical view.",
        cited_evidence_ids=("unknown-id",),
    )

    assert (
        build_revision_proposals(
            anchor=anchor,
            reports=reports,
            structure=structure,
            view=view,
            boundary_challenge=challenge,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("team_ids", "valid"),
    (
        ((), False),
        (("A",), False),
        (("A", "A"), False),
        (("A", "C"), False),
        (("A", "B"), True),
    ),
)
def test_boundary_challenge_requires_exact_canonical_dual_team_context(
    team_ids: tuple[str, ...],
    valid: bool,
) -> None:
    source = _reports_with_b_only_challenger()
    reports = tuple(
        source[index].model_copy(update={"team_id": team_id})
        for index, team_id in enumerate(team_ids)
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.PASS,
        rationale="A challenge exists only after canonical teams A and B complete.",
    )

    errors = boundary_challenge_semantic_errors(challenge, reports, _view())
    proposals = build_revision_proposals(
        anchor=_anchor(),
        reports=reports,
        structure=_structure(),
        view=_view(),
        boundary_challenge=challenge,
    )

    if valid:
        assert errors == ()
        assert len(proposals) == 1
    else:
        assert errors == (
            "Boundary challenger requires exactly one canonical report "
            "from teams A and B.",
        )
        assert proposals == ()


def test_wrong_dimension_review_fails_closed_at_proposal_trust_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, reports, structure, view, repair_id, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.SYMPTOM_REVIEW,
        rationale="A root-owned repair cannot authorize a symptom review.",
        cited_evidence_ids=(repair_id,),
    )

    assert (
        build_revision_proposals(
            anchor=anchor,
            reports=reports,
            structure=structure,
            view=view,
            boundary_challenge=challenge,
        )
        == ()
    )


def test_unowned_sealed_repair_challenge_cannot_expand_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, reports, structure, view, repair_id, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None
    reports = tuple(
        report.model_copy(
            update={
                "root_cause": report.root_cause.model_copy(
                    update={
                        "supporting_evidence_ids": (
                            report.symptom.supporting_evidence_ids[0],
                        ),
                        "counter_evidence_ids": (
                            report.symptom.supporting_evidence_ids[0],
                        ),
                        "boundary_evidence_ids": (),
                    }
                )
            }
        )
        for report in reports
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.CAUSE_REVIEW,
        rationale="Sealed authority is insufficient without canonical team ownership.",
        cited_evidence_ids=(repair_id,),
    )

    assert (
        build_revision_proposals(
            anchor=anchor,
            reports=reports,
            structure=structure,
            view=view,
            boundary_challenge=challenge,
        )
        == ()
    )


@pytest.mark.parametrize(
    "mutation",
    ("removed_authority", "counter_only", "wrong_dimension", "caller_copy"),
)
def test_repair_backed_alternative_fails_closed_without_exact_sealed_authority(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    capabilities = (
        ("symptom_observation",)
        if mutation == "wrong_dimension"
        else ("defect_mechanism",)
    )
    authority = (
        EvidenceAuthority.COUNTER_ONLY
        if mutation == "counter_only"
        else EvidenceAuthority.DIRECT
    )
    anchor, reports, structure, view, _, runtime = _sealed_candidate_context(
        monkeypatch,
        repair_capabilities=capabilities,
        repair_authority=authority,
    )
    assert runtime.ledger.view() is not None
    if mutation == "removed_authority":
        reports = tuple(
            report.model_copy(
                update={
                    "root_cause": report.root_cause.model_copy(
                        update={"boundary_evidence_ids": ()}
                    )
                }
            )
            for report in reports
        )
    elif mutation == "caller_copy":
        view = view.model_copy(deep=True)

    assert (
        build_revision_proposals(
            anchor=anchor,
            reports=reports,
            structure=structure,
            view=view,
        )
        == ()
    )


def test_candidate_ranking_is_stable_and_bounded_to_two_per_dimension() -> None:
    root_labels = (
        "Caller Contract Violation",
        "Internal Branch Defect",
        "Dependency Defect",
        "State Management Defect",
    )
    view = _view().model_copy(
        update={
            "taxonomy": {
                "symptom": _view().taxonomy["symptom"],
                "root_cause": root_labels,
            }
        }
    )
    structure = _structure().model_copy(
        update={
            "nodes": (
                *_structure().nodes,
                TaxonomyNode(
                    dimension="root_cause",
                    label="Dependency Defect",
                    definition_semantic_origin=TaxonomySemanticOrigin.PAPER_DEFINITION,
                    structure_semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                    definition="Official definition of Dependency Defect.",
                    abstraction_level="classification label",
                    responsibility_scope="synthetic component",
                    concept_kind="mechanism",
                    nearest_neighbors=("Caller Contract Violation",),
                ),
                TaxonomyNode(
                    dimension="root_cause",
                    label="State Management Defect",
                    definition_semantic_origin=TaxonomySemanticOrigin.PAPER_DEFINITION,
                    structure_semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                    definition="Official definition of State Management Defect.",
                    abstraction_level="classification label",
                    responsibility_scope="synthetic component",
                    concept_kind="mechanism",
                    nearest_neighbors=("Caller Contract Violation",),
                ),
            ),
            "boundary_cards": (
                *_structure().boundary_cards,
                _card(
                    "dependency-boundary",
                    "root_cause",
                    "Caller Contract Violation",
                    "Dependency Defect",
                ),
                _card(
                    "state-boundary",
                    "root_cause",
                    "Caller Contract Violation",
                    "State Management Defect",
                ),
            ),
        }
    )
    reports = (
        _report("A", root_label="Internal Branch Defect", root_evidence="cause-a"),
        _report("B", root_evidence="cause-b").model_copy(
            update={
                "root_cause": _root("Caller Contract Violation", "cause-b").model_copy(
                    update={
                        "label": "Dependency Defect",
                        "alternative_label": "Caller Contract Violation",
                    }
                )
            }
        ),
    )

    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=structure, view=view
    )
    reversed_proposals = build_revision_proposals(
        anchor=_anchor(),
        reports=tuple(reversed(reports)),
        structure=structure,
        view=view,
    )

    assert [proposal.proposed_label for proposal in proposals] == [
        "Dependency Defect",
        "Internal Branch Defect",
    ]
    assert proposals == reversed_proposals


@pytest.mark.parametrize("invalid_card_mode", ("missing", "duplicate"))
def test_ineligible_high_rank_signal_does_not_consume_exact_card_top_two(
    monkeypatch: pytest.MonkeyPatch,
    invalid_card_mode: str,
) -> None:
    labels = (
        "Caller Contract Violation",
        "Internal Branch Defect",
        "Dependency Defect",
        "State Management Defect",
    )
    anchor, reports, structure, view, repair_id, runtime = _sealed_candidate_context(
        monkeypatch,
        root_labels=labels,
        root_card_labels=(
            ("Internal Branch Defect", "Dependency Defect")
            if invalid_card_mode == "missing"
            else (
                "Internal Branch Defect",
                "Dependency Defect",
                "State Management Defect",
                "State Management Defect",
            )
        ),
    )
    assert runtime.ledger.view() is not None
    reports = (
        reports[0].model_copy(
            update={
                "root_cause": reports[0].root_cause.model_copy(
                    update={
                        "label": "Internal Branch Defect",
                        "alternative_label": "State Management Defect",
                        "boundary_evidence_ids": (repair_id,),
                    }
                )
            }
        ),
        reports[1].model_copy(
            update={
                "root_cause": reports[1].root_cause.model_copy(
                    update={
                        "label": "Dependency Defect",
                        "alternative_label": "Caller Contract Violation",
                        "boundary_evidence_ids": (),
                    }
                )
            }
        ),
    )

    proposals = build_revision_proposals(
        anchor=anchor, reports=reports, structure=structure, view=view
    )

    assert [proposal.proposed_label for proposal in proposals] == [
        "Dependency Defect",
        "Internal Branch Defect",
    ]


def test_caller_cannot_invent_verifier_signal_with_noncanonical_report_copy() -> None:
    verifier = DimensionVerificationReport(
        dimension=EvidenceDimension.ROOT_CAUSE,
        verdict=VerificationVerdict.REJECT,
        anchor_label="Caller Contract Violation",
        alternative_label="Internal Branch Defect",
        rationale="A caller-mutated report must not create verifier provenance.",
        supporting_evidence_ids=("cause-a",),
        corrected_claim="The caller-mutated report claims another mechanism.",
        corrected_causal_chain=("Input arrives.", "Logic mutates.", "Output fails."),
        confidence=0.8,
    )
    audit = TeamCorrectionAudit(
        dimension=EvidenceDimension.ROOT_CAUSE,
        anchor_label="Caller Contract Violation",
        proposed_label="Internal Branch Defect",
        final_label="Internal Branch Defect",
        accepted=True,
        reason="This audit was injected without the required anchored report.",
        supporting_evidence_ids=("cause-a",),
    )
    forged = _report("A").model_copy(
        update={"verifications": (verifier,), "correction_audit": (audit,)}
    )

    assert (
        build_revision_proposals(
            anchor=_anchor(),
            reports=(forged, _report("B")),
            structure=_structure(),
            view=_view(),
        )
        == ()
    )


def test_challenge_provenance_is_required_to_rebuild_routed_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, reports, structure, view, repair_id, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None
    reports = tuple(
        report.model_copy(
            update={
                "root_cause": report.root_cause.model_copy(
                    update={
                        "boundary_evidence_ids": (),
                        "counter_evidence_ids": (repair_id,),
                    }
                )
            }
        )
        for report in reports
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.CAUSE_REVIEW,
        rationale="Route only the already nominated root-cause alternative.",
        cited_evidence_ids=(repair_id,),
    )
    proposal = build_revision_proposals(
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
        boundary_challenge=challenge,
    )[0]

    assert revision_proposal_matches_context(
        proposal,
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
        boundary_challenge=challenge,
    )
    assert not revision_proposal_matches_context(
        proposal,
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
    )


@pytest.mark.parametrize(
    "field",
    ("source_kind", "report_digest", "alternative_label", "repair_evidence_id"),
)
def test_forged_candidate_signal_never_matches_canonical_context(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    anchor, reports, structure, view, _, runtime = _sealed_candidate_context(
        monkeypatch
    )
    assert runtime.ledger.view() is not None
    proposal = build_revision_proposals(
        anchor=anchor, reports=reports, structure=structure, view=view
    )[0]
    signal = proposal.candidate_signals[0]
    updates: dict[str, object]
    if field == "source_kind":
        updates = {"source_kinds": (RevisionCandidateSourceKind.TEAM_FINAL,)}
    elif field == "report_digest":
        updates = {"report_digests": ("0" * 64, "1" * 64)}
    elif field == "alternative_label":
        updates = {"label": "Caller Contract Violation"}
    else:
        updates = {
            "authoritative_repair_evidence_ids": (),
        }
    forged = proposal.model_copy(
        update={"candidate_signals": (signal.model_copy(update=updates),)}
    )

    assert not revision_proposal_matches_context(
        forged,
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
    )


def _candidate_bound_assessment(
    team_id: str, proposal: RevisionProposalEnvelope
) -> BaselineRevisionAssessment:
    return BaselineRevisionAssessment(
        assessor_team_id=team_id,
        proposal_digest=revision_proposal_digest(proposal),
        dimension=proposal.dimension,
        verdict=RevisionAssessmentVerdict.REVISE,
        baseline_label=proposal.baseline_label,
        proposed_label=proposal.proposed_label,
        boundary_card_id=proposal.boundary_card_id,
        contradicted_baseline_condition=(
            "evidence instead satisfies Internal Branch Defect"
        ),
        satisfied_proposed_conditions=("evidence satisfies Internal Branch Defect",),
        supporting_evidence_ids=("cause-b",),
        counter_evidence_ids=("cause-b",),
        baseline_source_config_hash=proposal.baseline_source_config_hash,
        baseline_source_predictions_sha256=(
            proposal.baseline_source_predictions_sha256
        ),
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
    )


def _cross_check(
    checker_team_id: str,
    assessment: BaselineRevisionAssessment,
) -> RevisionConsistencyReport:
    assert assessment.assessor_team_id is not None
    assert assessment.proposal_digest is not None
    assert assessment.baseline_label is not None
    assert assessment.proposed_label is not None
    assert assessment.boundary_card_id is not None
    return RevisionConsistencyReport(
        checker_team_id=checker_team_id,
        assessment_owner_team_id=assessment.assessor_team_id,
        proposal_digest=assessment.proposal_digest,
        dimension=assessment.dimension,
        baseline_label=assessment.baseline_label,
        proposed_label=assessment.proposed_label,
        status=ConsistencyStatus.CONSISTENT,
        rationale="The opposite perspective independently confirms the exact boundary.",
        supporting_evidence_ids=assessment.supporting_evidence_ids,
        counter_evidence_ids=assessment.counter_evidence_ids,
        assessment_digest=baseline_revision_assessment_digest(assessment),
        boundary_card_id=assessment.boundary_card_id,
        baseline_source_config_hash=assessment.baseline_source_config_hash,
        baseline_source_predictions_sha256=(
            assessment.baseline_source_predictions_sha256
        ),
        taxonomy_structure_hash=assessment.taxonomy_structure_hash,
        evidence_view_hash=assessment.evidence_view_hash,
    )


def test_certificate_uses_dual_cross_checks_for_b_only_proposal() -> None:
    """Raw A label may stay Baseline; both candidate-bound reviews own authority."""

    reports = _reports_with_b_only_challenger()
    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=_view()
    )
    assert len(proposals) == 1
    assessment_a = _candidate_bound_assessment("A", proposals[0])
    assessment_b = _candidate_bound_assessment("B", proposals[0])
    augmented = (
        reports[0].model_copy(
            update={
                "baseline_revision_assessments": (assessment_a,),
                "revision_consistency": (_cross_check("A", assessment_b),),
            }
        ),
        reports[1].model_copy(
            update={
                "baseline_revision_assessments": (assessment_b,),
                "revision_consistency": (_cross_check("B", assessment_a),),
            }
        ),
    )

    certificates = compose_baseline_revision_certificates(
        anchor=_anchor(),
        reports=augmented,
        proposals=proposals,
        structure=_structure(),
        view=_view(),
    )

    assert len(certificates) == 1
    assert certificates[0].proposed_label == "Internal Branch Defect"
    assert certificates[0].proposal_digest == revision_proposal_digest(proposals[0])


def _frozen_revision_view(*, authoritative: bool) -> EvidenceView:
    view = _view()
    frozen_cause = next(item for item in view.items if item.evidence_id == "cause-b")
    frozen_cause = frozen_cause.model_copy(
        update={
            "metadata": {
                "frozen_evidence": True,
                "frozen_node_type": (
                    "changed_code" if authoritative else "linked_issue"
                ),
                "frozen_authority": "direct" if authoritative else "context_only",
                "citation_usage": (
                    "support_or_counter" if authoritative else "context_only"
                ),
                "evidence_capabilities": ("defect_mechanism",),
            }
        }
    )
    return view.model_copy(
        update={
            "items": tuple(
                frozen_cause if item.evidence_id == "cause-b" else item
                for item in view.items
            )
        }
    )


def _candidate_bound_certificates_for_view(
    view: EvidenceView,
) -> tuple[object, ...]:
    reports = _reports_with_b_only_challenger()
    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=view
    )
    assert len(proposals) == 1
    assessment_a = _candidate_bound_assessment("A", proposals[0])
    assessment_b = _candidate_bound_assessment("B", proposals[0])
    augmented = (
        reports[0].model_copy(
            update={
                "baseline_revision_assessments": (assessment_a,),
                "revision_consistency": (_cross_check("A", assessment_b),),
            }
        ),
        reports[1].model_copy(
            update={
                "baseline_revision_assessments": (assessment_b,),
                "revision_consistency": (_cross_check("B", assessment_a),),
            }
        ),
    )
    return compose_baseline_revision_certificates(
        anchor=_anchor(),
        reports=augmented,
        proposals=proposals,
        structure=_structure(),
        view=view,
    )


def test_frozen_context_only_issue_cannot_authorize_revision_certificate() -> None:
    assert (
        _candidate_bound_certificates_for_view(
            _frozen_revision_view(authoritative=False)
        )
        == ()
    )


def test_neighbor_case_cannot_be_used_as_positive_revision_support() -> None:
    view = _frozen_revision_view(authoritative=False)
    neighbor = next(item for item in view.items if item.evidence_id == "cause-b")
    neighbor = neighbor.model_copy(
        update={
            "metadata": {
                **dict(neighbor.metadata),
                "frozen_node_type": "neighbor_case",
                "frozen_authority": "counter_only",
                "citation_usage": "counter_only",
            }
        }
    )
    view = view.model_copy(
        update={
            "items": tuple(
                neighbor if item.evidence_id == "cause-b" else item
                for item in view.items
            )
        }
    )

    assert (
        build_revision_proposals(
            anchor=_anchor(),
            reports=_reports_with_b_only_challenger(),
            structure=_structure(),
            view=view,
        )
        == ()
    )


def test_preservation_gate_rejects_forged_context_only_revision_certificate() -> None:
    view = _frozen_revision_view(authoritative=False)
    reports = _reports_with_b_only_challenger()
    proposal = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=view
    )[0]
    forged = LabelRevisionCertificate(
        dimension=EvidenceDimension.ROOT_CAUSE,
        proposal_digest=revision_proposal_digest(proposal),
        baseline_label="Caller Contract Violation",
        proposed_label="Internal Branch Defect",
        revision_basis=RevisionBasis.TAXONOMY_BOUNDARY,
        boundary_card_id="cause-boundary",
        contradicted_baseline_condition=(
            "evidence instead satisfies Internal Branch Defect"
        ),
        satisfied_proposed_conditions=("evidence satisfies Internal Branch Defect",),
        supporting_evidence_ids=("cause-b",),
        counter_evidence_ids=("cause-b",),
        supporting_team_ids=("A", "B"),
    )

    preservation = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=None,
        candidate_verification=None,
        proposals=(proposal,),
        certificates=(forged,),
        structure=_structure(),
        view=view,
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
        expected_anchor_hash=baseline_anchor_hash(_anchor()),
        expected_structure_hash=taxonomy_structure_hash(_structure()),
    )

    assert preservation.root_cause_label == "Caller Contract Violation"
    assert preservation.applied_certificates == ()


def test_self_signed_frozen_repair_mechanism_cannot_authorize_revision_certificate() -> (
    None
):
    certificates = _candidate_bound_certificates_for_view(
        _frozen_revision_view(authoritative=True)
    )

    assert certificates == ()


def test_certificate_rejects_self_check_for_candidate_bound_proposal() -> None:
    reports = _reports_with_b_only_challenger()
    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=_view()
    )
    assessment_a = _candidate_bound_assessment("A", proposals[0])
    assessment_b = _candidate_bound_assessment("B", proposals[0])
    augmented = (
        reports[0].model_copy(
            update={
                "baseline_revision_assessments": (assessment_a,),
                # A illegally checks its own assessment instead of B's.
                "revision_consistency": (_cross_check("B", assessment_a),),
            }
        ),
        reports[1].model_copy(
            update={
                "baseline_revision_assessments": (assessment_b,),
                "revision_consistency": (_cross_check("A", assessment_b),),
            }
        ),
    )

    assert (
        compose_baseline_revision_certificates(
            anchor=_anchor(),
            reports=augmented,
            proposals=proposals,
            structure=_structure(),
            view=_view(),
        )
        == ()
    )


def test_gate_applies_certificate_owned_candidate_when_pre_gate_kept_baseline() -> None:
    reports = _reports_with_b_only_challenger()
    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=_view()
    )
    assessment_a = _candidate_bound_assessment("A", proposals[0])
    assessment_b = _candidate_bound_assessment("B", proposals[0])
    augmented = (
        reports[0].model_copy(
            update={
                "baseline_revision_assessments": (assessment_a,),
                "revision_consistency": (_cross_check("A", assessment_b),),
            }
        ),
        reports[1].model_copy(
            update={
                "baseline_revision_assessments": (assessment_b,),
                "revision_consistency": (_cross_check("B", assessment_a),),
            }
        ),
    )
    certificates = compose_baseline_revision_certificates(
        anchor=_anchor(),
        reports=augmented,
        proposals=proposals,
        structure=_structure(),
        view=_view(),
    )
    pre_gate = Stage3FinalDecision(
        symptom_label="Returned Anomaly",
        root_cause_label="Caller Contract Violation",
        confidence=0.83,
        rationale="The ordinary pre-gate path retained both frozen Baseline labels.",
        supporting_evidence_ids=("symptom-a", "cause-a"),
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )

    preservation = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=pre_gate,
        candidate_verification=None,
        proposals=proposals,
        certificates=certificates,
        structure=_structure(),
        view=_view(),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
        expected_anchor_hash=baseline_anchor_hash(_anchor()),
        expected_structure_hash=taxonomy_structure_hash(_structure()),
    )
    final = compose_baseline_preservation_decision(
        preservation=preservation,
        candidate=pre_gate,
    )

    assert preservation.root_cause_label == "Internal Branch Defect"
    assert preservation.symptom_label == "Returned Anomaly"
    assert final.baseline_gate_provenance is BaselineGateProvenance.REVISION_CERTIFICATE
    assert final.confidence is None
    assert final.supporting_evidence_ids == ("cause-b",)


@pytest.mark.parametrize("forged_digest", ("f" * 64, None))
def test_candidate_bound_certificate_cannot_fall_back_to_legacy_candidate_path(
    forged_digest: str | None,
) -> None:
    reports = _reports_with_b_only_challenger()
    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=_view()
    )
    assessment_a = _candidate_bound_assessment("A", proposals[0])
    assessment_b = _candidate_bound_assessment("B", proposals[0])
    augmented = (
        reports[0].model_copy(
            update={
                "baseline_revision_assessments": (assessment_a,),
                "revision_consistency": (_cross_check("A", assessment_b),),
            }
        ),
        reports[1].model_copy(
            update={
                "baseline_revision_assessments": (assessment_b,),
                "revision_consistency": (_cross_check("B", assessment_a),),
            }
        ),
    )
    certificate = compose_baseline_revision_certificates(
        anchor=_anchor(),
        reports=augmented,
        proposals=proposals,
        structure=_structure(),
        view=_view(),
    )[0].model_copy(update={"proposal_digest": forged_digest})
    supplied_proposals = proposals if forged_digest is not None else ()
    matching_candidate = Stage3FinalDecision(
        symptom_label="Returned Anomaly",
        root_cause_label="Internal Branch Defect",
        confidence=0.81,
        rationale="A matching ordinary candidate must not replace proposal authority.",
        supporting_evidence_ids=("cause-b",),
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )

    with pytest.raises(ValueError, match="proposal"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=matching_candidate,
            candidate_verification=None,
            proposals=supplied_proposals,
            certificates=(certificate,),
            structure=_structure(),
            view=_view(),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
            expected_anchor_hash=baseline_anchor_hash(_anchor()),
            expected_structure_hash=taxonomy_structure_hash(_structure()),
        )


def test_proposals_include_single_team_challenger_without_pre_gate_input() -> None:
    proposals = build_revision_proposals(
        anchor=_anchor(),
        reports=_reports_with_b_only_challenger(),
        structure=_structure(),
        view=_view(),
    )

    assert [
        (
            proposal.dimension,
            proposal.baseline_label,
            proposal.proposed_label,
            proposal.proposer_team_ids,
        )
        for proposal in proposals
    ] == [
        (
            EvidenceDimension.ROOT_CAUSE,
            "Caller Contract Violation",
            "Internal Branch Defect",
            ("B",),
        )
    ]


def test_unreferenced_quarantined_item_does_not_erase_a_valid_proposal() -> None:
    view = _view().model_copy(
        update={
            "items": (
                *_view().items,
                _item("unreferenced", "untrusted_source"),
            )
        }
    )

    proposals = build_revision_proposals(
        anchor=_anchor(),
        reports=_reports_with_b_only_challenger(),
        structure=_structure(),
        view=view,
    )

    assert len(proposals) == 1


def test_same_challenger_is_merged_with_canonical_teams_digests_and_citations() -> None:
    reports = (
        _report("B", root_label="Internal Branch Defect", root_evidence="cause-b"),
        _report("A", root_label="Internal Branch Defect", root_evidence="cause-a"),
    )

    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=_view()
    )
    reversed_proposals = build_revision_proposals(
        anchor=_anchor(),
        reports=tuple(reversed(reports)),
        structure=_structure(),
        view=_view(),
    )

    assert len(proposals) == 1
    assert proposals == reversed_proposals
    assert proposals[0].proposer_team_ids == ("A", "B")
    assert len(proposals[0].proposer_report_digests) == 2
    assert proposals[0].supporting_evidence_ids == ("cause-a", "cause-b")
    assert revision_proposal_digest(proposals[0]) == revision_proposal_digest(
        reversed_proposals[0]
    )


def test_classification_report_digest_excludes_revision_owned_appendages() -> None:
    report = _report("B", root_label="Internal Branch Defect", root_evidence="cause-b")
    assessment = BaselineRevisionAssessment(
        dimension=EvidenceDimension.ROOT_CAUSE,
        verdict=RevisionAssessmentVerdict.PRESERVE,
        baseline_source_config_hash="a" * 64,
        baseline_source_predictions_sha256="b" * 64,
        taxonomy_structure_hash="c" * 64,
        evidence_view_hash="d" * 64,
    )
    augmented = report.model_copy(
        update={
            "baseline_revision_assessments": (assessment,),
            "revision_consistency": (
                RevisionConsistencyReport(
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    baseline_label="Caller Contract Violation",
                    proposed_label="Internal Branch Defect",
                    status=ConsistencyStatus.CONSISTENT,
                    rationale="The frozen classification remains causally coherent after review.",
                    supporting_evidence_ids=("cause-b",),
                    assessment_digest="e" * 64,
                    boundary_card_id="cause-boundary",
                    baseline_source_config_hash="a" * 64,
                    baseline_source_predictions_sha256="b" * 64,
                    taxonomy_structure_hash="c" * 64,
                    evidence_view_hash="d" * 64,
                ),
            ),
        }
    )
    changed_classification = report.model_copy(
        update={
            "root_cause": report.root_cause.model_copy(
                update={
                    "defect_mechanism": "A different frozen mechanism changes the classification report."
                }
            )
        }
    )

    assert stage3_team_report_digest(augmented) == stage3_team_report_digest(report)
    assert stage3_team_report_digest(
        changed_classification
    ) != stage3_team_report_digest(report)
    assert build_revision_proposals(
        anchor=_anchor(),
        reports=(_report("A"), augmented),
        structure=_structure(),
        view=_view(),
    ) == build_revision_proposals(
        anchor=_anchor(),
        reports=(_report("A"), report),
        structure=_structure(),
        view=_view(),
    )


def test_distinct_challengers_have_dimension_then_label_stable_order() -> None:
    reports = (
        _report("A", symptom_label="Execution Halt", symptom_evidence="symptom-a"),
        _report("B", root_label="Internal Branch Defect", root_evidence="cause-b"),
    )

    proposals = build_revision_proposals(
        anchor=_anchor(), reports=reports, structure=_structure(), view=_view()
    )

    assert [
        (proposal.dimension, proposal.proposed_label) for proposal in proposals
    ] == [
        (EvidenceDimension.SYMPTOM, "Execution Halt"),
        (EvidenceDimension.ROOT_CAUSE, "Internal Branch Defect"),
    ]


@pytest.mark.parametrize(
    "update",
    [
        {"baseline_label": ""},
        {"proposed_label": ""},
        {"proposed_label": "Caller Contract Violation"},
        {"proposer_team_ids": ("B", "A")},
        {"proposer_report_digests": ("c" * 64, "c" * 64)},
        {"supporting_evidence_ids": ()},
    ],
)
def test_envelope_rejects_incomplete_or_noncanonical_identity(
    update: dict[str, object],
) -> None:
    payload = {
        "dimension": EvidenceDimension.ROOT_CAUSE,
        "baseline_label": "Caller Contract Violation",
        "proposed_label": "Internal Branch Defect",
        "boundary_card_id": "cause-boundary",
        "proposer_team_ids": ("A",),
        "proposer_report_digests": ("c" * 64,),
        "supporting_evidence_ids": ("cause-a",),
        "counter_evidence_ids": (),
        "baseline_source_config_hash": "a" * 64,
        "baseline_source_predictions_sha256": "b" * 64,
        "taxonomy_structure_hash": "d" * 64,
        "evidence_view_hash": "e" * 64,
    }
    payload.update(update)

    with pytest.raises(ValidationError):
        RevisionProposalEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "invalid_baseline",
        "record_mismatch",
        "domain_mismatch",
        "taxonomy_mismatch",
        "missing_card",
        "wrong_card_pair",
        "unknown_citation",
        "incapable_citation",
    ],
)
def test_invalid_authority_or_candidate_citations_fail_closed(mutation: str) -> None:
    anchor = _anchor()
    reports = _reports_with_b_only_challenger()
    structure = _structure()
    view = _view()
    if mutation == "invalid_baseline":
        anchor = _anchor(valid=False)
    elif mutation == "record_mismatch":
        anchor = anchor.model_copy(update={"record_id": "other-record"})
    elif mutation == "domain_mismatch":
        structure = structure.model_copy(update={"domain": "other-domain"})
    elif mutation == "taxonomy_mismatch":
        view = view.model_copy(
            update={
                "taxonomy": {
                    "symptom": ["Returned Anomaly", "Execution Halt"],
                    "root_cause": ["Caller Contract Violation"],
                }
            }
        )
    elif mutation == "missing_card":
        structure = structure.model_copy(update={"boundary_cards": ()})
    elif mutation == "wrong_card_pair":
        structure = structure.model_copy(
            update={"boundary_cards": (structure.boundary_cards[0],)}
        )
    elif mutation == "unknown_citation":
        reports = (
            reports[0],
            _report("B", root_label="Internal Branch Defect", root_evidence="missing"),
        )
    else:
        reports = (
            reports[0],
            _report(
                "B",
                root_label="Internal Branch Defect",
                root_evidence="symptom-only",
            ),
        )

    assert (
        build_revision_proposals(
            anchor=anchor, reports=reports, structure=structure, view=view
        )
        == ()
    )


@pytest.mark.parametrize(
    "field",
    [
        "boundary_card_id",
        "proposer_report_digests",
        "supporting_evidence_ids",
        "baseline_source_config_hash",
        "baseline_source_predictions_sha256",
        "taxonomy_structure_hash",
        "evidence_view_hash",
    ],
)
def test_forged_envelope_never_matches_canonical_context(field: str) -> None:
    anchor = _anchor()
    reports = _reports_with_b_only_challenger()
    structure = _structure()
    view = _view()
    proposal = build_revision_proposals(
        anchor=anchor, reports=reports, structure=structure, view=view
    )[0]
    if field == "boundary_card_id":
        update: object = "symptom-boundary"
    elif field == "proposer_report_digests":
        update = ("0" * 64,)
    elif field == "supporting_evidence_ids":
        update = ("cause-a",)
    else:
        update = "0" * 64
    forged = proposal.model_copy(update={field: update})

    assert not revision_proposal_matches_context(
        forged,
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
    )
    assert proposal.taxonomy_structure_hash == taxonomy_structure_hash(structure)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_uri", "https://changed.example.test/cause-b"),
        ("retrieved_at", "2026-08-17T00:00:00Z"),
        ("explicitness", EvidenceExplicitness.INFERRED),
    ],
)
def test_evidence_identity_mutation_invalidates_proposal_context(
    field: str, value: object
) -> None:
    anchor = _anchor()
    reports = _reports_with_b_only_challenger()
    structure = _structure()
    view = _view()
    proposal = build_revision_proposals(
        anchor=anchor, reports=reports, structure=structure, view=view
    )[0]
    changed_items = tuple(
        (
            item.model_copy(update={field: value})
            if item.evidence_id == "cause-b"
            else item
        )
        for item in view.items
    )
    changed_view = view.model_copy(update={"items": changed_items})

    assert not revision_proposal_matches_context(
        proposal,
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=changed_view,
    )
