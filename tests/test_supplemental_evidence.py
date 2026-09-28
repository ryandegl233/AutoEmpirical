"""Catch lost pixels, cross-case attachments, and answer-bearing bundle fields."""
import base64
import hashlib
import importlib
import io
import json

import pytest
from PIL import Image

from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceItem


def bundle_class():
    try:
        return importlib.import_module(
            'Benchmark.src.adaptive_empirical_workflow.supplemental_evidence'
        ).SupplementalEvidenceBundle
    except ModuleNotFoundError:
        pytest.fail('Supplemental evidence and original-image transport are not implemented')


def fixture_bundle(tmp_path):
    picture = io.BytesIO()
    Image.new('RGB', (3, 2), (19, 67, 131)).save(picture, format='PNG')
    raw = picture.getvalue()
    path = tmp_path / 'input.png'
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    content = 'Original attachment. Its pixels accompany this item in model requests.'
    item = dict(evidence_id='extra-image-one', record_id='case-one',
                source_type='runtime_observation', source_uri='https://example.org/attachment.png',
                retrieved_at='2026-09-22T00:00:00Z', content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness='direct', metadata={'supplemental_image_sha256': sha})
    payload = {'schema_version': 1, 'information_policy': 'public_at_collection',
               'records': [{'record_id': 'case-one', 'items': [item], 'images': [
                   {'evidence_id': item['evidence_id'], 'path': 'input.png',
                    'sha256': sha, 'mime_type': 'image/png'}]}]}
    manifest = tmp_path / 'bundle.json'
    manifest.write_text(json.dumps(payload), encoding='utf-8')
    return manifest, item, raw, payload


def test_pixels_are_attached_to_visible_evidence_and_survive_schema_repair(tmp_path):
    manifest, item, raw, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one','case-two'})
    prompt = json.dumps({'evidence_view': {'record_id': 'case-one', 'items': [item]}})
    prompt += '\nRepair the output schema only.'
    content, attached = bundle.prepare_message(prompt)
    assert content[0] == {'type': 'text', 'text': prompt}
    image_parts = [p for p in content if p['type'] == 'image_url']
    assert len(image_parts) == 1
    assert base64.b64decode(image_parts[0]['image_url']['url'].split(',',1)[1]) == raw
    assert attached[0]['record_id'] == 'case-one'
    assert [x.evidence_id for x in bundle.items_for('case-one')] == ['extra-image-one']
    assert bundle.items_for('case-two') == ()


def test_mentioning_an_image_id_without_its_evidence_never_attaches_it(tmp_path):
    manifest, item, _, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one','case-two'})
    prompt = json.dumps({'evidence_view': {'record_id':'case-two','items':[]},
                         'context': {'prior_report': item['evidence_id']}})
    assert bundle.prepare_message(prompt) == (prompt, ())


def test_arbitration_gets_only_images_in_its_bounded_snapshot(tmp_path):
    manifest, item, _, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one'})
    prompt = json.dumps({'evidence_snapshot': {'items': [item]}, 'candidates': []})
    content, images = bundle.prepare_message(prompt)
    assert isinstance(content, list)
    assert [i['evidence_id'] for i in images] == ['extra-image-one']


@pytest.mark.parametrize('mutation', ['record_mismatch', 'gold_metadata', 'extra_root_field', 'unknown_case'])
def test_rejects_cross_case_or_answer_bearing_bundles(tmp_path, mutation):
    manifest, _, _, payload = fixture_bundle(tmp_path)
    if mutation == 'record_mismatch': payload['records'][0]['items'][0]['record_id'] = 'case-two'
    if mutation == 'gold_metadata': payload['records'][0]['items'][0]['metadata']['gold_label'] = 'answer'
    if mutation == 'extra_root_field': payload['GT'] = 'answer'
    if mutation == 'unknown_case': payload['records'][0]['record_id'] = 'outside-cohort'
    manifest.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        bundle_class().load(manifest, allowed_record_ids={'case-one'})


def test_image_modified_after_loading_is_rejected_before_request(tmp_path):
    manifest, item, _, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one'})
    (tmp_path/'input.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        bundle.prepare_message(json.dumps({'evidence_view': {'record_id':'case-one','items':[item]}}))


def test_rejects_altered_item_under_valid_image_id(tmp_path):
    manifest, item, _, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one'})
    item['record_id'] = 'case-two'
    with pytest.raises(ValueError):
        bundle.prepare_message(json.dumps({'evidence_view': {'record_id':'case-two','items':[item]}}))


def test_transport_accounting_is_scoped_to_case_and_records_failures(tmp_path):
    manifest, item, _, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one','case-two'})
    _, images = bundle.prepare_message(json.dumps({'evidence_view': {'items':[item]}}))
    bundle.record_delivery(images, role='joint_anchor', status='failed')
    audit = bundle.audit_for('case-one')
    assert audit['image_request_events'][0]['status'] == 'failed'
    assert bundle.audit_for('case-two')['image_request_events'] == []


def test_supplement_is_in_the_ledger_before_readiness_and_in_final_audit(tmp_path):
    from Benchmark.src.adaptive_empirical_workflow.experiment import run_adaptive_record
    from tests.test_adaptive_empirical_workflow_experiment import _DeterministicAgents
    manifest, _, _, _ = fixture_bundle(tmp_path)
    bundle = bundle_class().load(manifest, allowed_record_ids={'case-one'})
    class ObservingAgents(_DeterministicAgents):
        def evidence_readiness(self, task, view):
            assert 'extra-image-one' in [i.evidence_id for i in view.items]
            return super().evidence_readiness(task, view)
    agents = ObservingAgents()
    row = run_adaptive_record({'record_id':'case-one','body':'A request crashes.',
                               'code_diff':'- reject(request)\n+ process(request)'},
        domain='ase2022', taxonomy={'symptom':['Crash','Poor Performance'],
        'root_cause':['Incorrect Code Logic','API Misuse']}, agents=agents, stage='stage3',
        supplemental_items=bundle.items_for('case-one'))
    assert agents.readiness_tasks
    assert 'extra-image-one' in [i['evidence_id'] for i in row['audit']['evidence_items']]


def test_runner_passes_pixels_to_provider_and_records_failed_interpretation(tmp_path):
    from Benchmark.scripts.run_adaptive_empirical_workflow import run_cli
    from tests.test_run_adaptive_empirical_workflow import _write_inputs
    cohort, taxonomy = _write_inputs(tmp_path, record_id='case-one')
    manifest, _, raw, _ = fixture_bundle(tmp_path)
    received = []
    def fake_provider(system, user, **kwargs):
        # Simulate an external API accepting the image but returning invalid schema.
        if isinstance(user, list):
            for part in user:
                if part['type'] == 'image_url':
                    received.append(base64.b64decode(part['image_url']['url'].split(',',1)[1]))
        return '{}'
    summary = run_cli(['--domain','ase2022','--stage','stage3','--provider','gemini',
                      '--model','gemini-3.7-flash','--cohort-path',str(cohort),
                      '--taxonomy-path',str(taxonomy),'--output-dir',str(tmp_path/'run'),
                      '--supplemental-evidence',str(manifest),'--max-schema-retries','0',
                      '--retry-delay-seconds','0','--no-progress'], transport_override=fake_provider)
    assert received and all(x == raw for x in received)
    row = json.loads(__import__('pathlib').Path(summary['predictions_path']).read_text('utf-8').splitlines()[0])
    assert not row['stage3_valid']
    assert row['audit']['supplemental_evidence']['image_request_events']
