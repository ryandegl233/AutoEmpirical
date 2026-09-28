"""Regression checks for source-code and original-image evidence admission."""
import hashlib

import pytest

from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
from Benchmark.src.adaptive_empirical_workflow.evidence_chain import Citation, validate_citations
from tests.test_adaptive_empirical_workflow_a_decision import decision_query
from tests.test_adaptive_empirical_workflow_evidence_chain import sample


def source_view(source_type, metadata=None):
    view, _ = sample()
    item = view.items[0].model_copy(update={'source_type': source_type, 'metadata': metadata or {}})
    return view.model_copy(update={'items': (item,)})


def test_root_only_source_code_atom_is_accepted_without_expanding_symptom_sources():
    view = source_view('source_code')
    query = decision_query(form='observed_connection')
    atom = query.atoms[0].model_copy(update={'kind': 'observation', 'citations': tuple(
        c.model_copy(update={'kind': 'observation'}) for c in query.atoms[0].citations)})
    query = query.model_copy(update={'atoms': (atom,)})
    result = RuleDecisionTools().validate_query(view, query)
    assert result['root_gate']['selected_label'] == 'API Misuse'
    with pytest.raises(ValueError, match='not eligible'):
        validate_citations(atom.citations, view, 'symptom')


def test_image_reading_is_admitted_but_never_certified_as_verbatim_text():
    sha = hashlib.sha256(b'original image binding').hexdigest()
    view = source_view('runtime_observation', {'supplemental_image_sha256': sha, 'source_raw_sha256': sha})
    citation = Citation(evidence_id='e', quote='Visible error text absent from the placeholder.', kind='observation')
    result = validate_citations((citation,), view, 'symptom')
    assert result[0]['quote_verification'] == 'unverified_visual_interpretation'
    assert result[0]['semantic_truth_verified'] is False


@pytest.mark.parametrize('source_type,metadata', [
    ('issue_body', {}),
    ('runtime_observation', {}),
    ('issue_body', {'supplemental_image_sha256': 'a' * 64}),
    ('runtime_observation', {'supplemental_image_sha256': 'not-a-hash'}),
    ('runtime_observation', {'supplemental_image_sha256': 'a' * 64, 'source_raw_sha256': 'b' * 64}),
])
def test_non_image_or_invalid_binding_cannot_bypass_verbatim_text(source_type, metadata):
    citation = Citation(evidence_id='e', quote='Invented text.', kind='observation')
    with pytest.raises(ValueError):
        validate_citations((citation,), source_view(source_type, metadata), 'root_cause')


def test_image_reference_cannot_cross_records_or_invent_source_assertions():
    view = source_view('runtime_observation', {'supplemental_image_sha256': 'a' * 64})
    for citation in (
        Citation(evidence_id='other-case-image', quote='Visible text.', kind='observation'),
        Citation(evidence_id='e', quote='A maintainer confirmed the cause.', kind='source_assertion'),
    ):
        with pytest.raises(ValueError):
            validate_citations((citation,), view, 'root_cause')


def test_text_citation_still_has_exact_text_audit():
    view, _ = sample()
    citation = Citation(evidence_id='e', quote='The output differs.', kind='source_assertion')
    result = validate_citations((citation,), view, 'root_cause')
    assert result[0]['quote_verification'] == 'exact_text'
    assert result[0]['semantic_truth_verified'] is False
