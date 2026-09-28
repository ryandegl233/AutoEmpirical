import csv
import json
from pathlib import Path

import pytest

from Benchmark.src import paper_benchmark as papers


def write_stage(path, rows):
    fields = ['record_id', 'paper_id', 'source_project', 'issue_url', 'title', 'body',
              'comments', 'state', 'created_at', 'symptom', 'root_cause', 'original_label_json']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fixture_sources(tmp_path, reverse=False):
    paper = 'icse2023_an_empirical_study_on_bugs'
    rows = [dict(record_id=f'{paper}:{i}', paper_id=paper, source_project='pytorch',
                 issue_url=f'https://github.com/pytorch/pytorch/issues/{i}',
                 title=f'Report {i}', body='Observed failure in tensor operation.',
                 symptom='', root_cause='', original_label_json='{}') for i in range(12)]
    stage2 = [dict(row) for row in rows[:6]]
    stage3 = [dict(row, symptom='Crash', root_cause='Logic error') for row in rows[:4]]
    if reverse:
        rows.reverse(); stage2.reverse(); stage3.reverse()
    return tuple(write_stage(tmp_path / f'stage{i}.csv', data)
                 for i, data in enumerate((rows, stage2, stage3), 1))


def prepare(tmp_path, output, **kwargs):
    paths = fixture_sources(tmp_path, kwargs.pop('reverse', False))
    return papers.prepare_paper_artifacts('icse2023', stage1_path=paths[0],
        stage2_path=paths[1], stage3_path=paths[2], output_dir=output,
        positives=3, negatives=3, seed=123, **kwargs)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def test_balanced_cohort_stage2_only_never_negative_and_stage3_aligned(tmp_path):
    paths = prepare(tmp_path, tmp_path/'out')
    rows = read_csv(paths['cohort'])
    assert len(rows) == 6
    assert sum(row['decision'] == 'accepted_fault' for row in rows) == 3
    assert all(int(row['record_id'].rsplit(':',1)[1]) >= 6 for row in rows
               if row['decision'] == 'rejected_candidate')
    assert {r['record_id'] for r in read_csv(paths['stage3_sample'])} == {
        r['record_id'] for r in rows if r['decision'] == 'accepted_fault'}


def test_sampling_independent_of_csv_order(tmp_path):
    a = prepare(tmp_path, tmp_path/'a')
    b = prepare(tmp_path, tmp_path/'b', reverse=True)
    assert Path(a['cohort']).read_bytes() == Path(b['cohort']).read_bytes()


def test_gt_fields_cannot_change_model_evidence_or_prompt():
    row = {'paper_id':'icse2023_an_empirical_study_on_bugs','title':'Observed crash',
           'body':'Fails on zero size tensors.', 'symptom':'SECRET_A',
           'root_cause':'SECRET_B','original_label_json':'SECRET_C','fix_type':'SECRET_D',
           'decision':'accepted_fault','source_file':'SECRET_E'}
    before = papers.build_user_prompt(row)
    changed = {**row, 'symptom':'OTHER','root_cause':'OTHER',
               'original_label_json':'OTHER','decision':'rejected_candidate'}
    assert before == papers.build_user_prompt(changed)
    assert 'SECRET' not in before
    assert set(papers.model_evidence_fields(row)) <= set(papers.SAFE_MODEL_FIELDS)


def test_duplicate_id_fails_before_any_output(tmp_path):
    paths = fixture_sources(tmp_path)
    rows = read_csv(paths[0]); rows.append(rows[0]); write_stage(paths[0], rows)
    with pytest.raises(ValueError, match='duplicate record_id'):
        papers.prepare_paper_artifacts('icse2023', stage1_path=paths[0],
            stage2_path=paths[1],stage3_path=paths[2],output_dir=tmp_path/'out',
            positives=3,negatives=3)
    assert not (tmp_path/'out'/'cohort.csv').exists()


def test_insufficient_pool_is_explicit_not_sampled_with_replacement(tmp_path):
    paths = fixture_sources(tmp_path)
    with pytest.raises(ValueError, match='eligible positives'):
        papers.prepare_paper_artifacts('icse2023',stage1_path=paths[0],
            stage2_path=paths[1],stage3_path=paths[2],output_dir=tmp_path/'out',
            positives=5,negatives=3)


def test_existing_output_is_not_overwritten(tmp_path):
    prepare(tmp_path, tmp_path/'out')
    before = (tmp_path/'out'/'cohort.csv').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(tmp_path, tmp_path/'out')
    assert before == (tmp_path/'out'/'cohort.csv').read_bytes()


def test_manifest_binds_sources_and_every_prompt(tmp_path):
    paths = prepare(tmp_path,tmp_path/'out')
    manifest = json.loads(Path(paths['manifest']).read_text(encoding='utf-8'))
    assert len(manifest['source_sha256']) >= 3
    assert len(manifest['record_ids']) == 6
    assert manifest['annotation_record_ids'] == [r['record_id'] for r in read_csv(paths['stage3_sample'])]
    for name in ('cohort','taxonomy','stage2_prompts','stage3_prompts','stage3_sample'):
        assert manifest['artifact_sha256'][name] == papers.sha256_file(paths[name])


def test_unknown_domain_rejected():
    with pytest.raises(ValueError,match='unsupported'):
        papers.get_paper_profile('unknown-paper')


def test_native_taxonomies_do_not_enumerate_case_descriptions():
    uav = papers.build_taxonomy('fse2021')
    assert uav['symptom'] == []
    assert uav['annotation_modes']['symptom'] == 'free_text'
    iot = papers.build_taxonomy('icse2021')
    assert iot['annotation_modes']['symptom'] == 'multi_label'
    assert 'Auth issues' in iot['symptom']
    assert not any(',' in x for x in iot['symptom'])
    assert papers.build_taxonomy('icse2022')['annotation_modes']['symptom'] == 'constant'


def test_frozen_cohort_reuse_preserves_its_evidence_version(tmp_path):
    original = prepare(tmp_path,tmp_path/'first')
    rows = read_csv(original['cohort'])
    rows[0]['body'] = 'Earlier frozen source text.'
    frozen = tmp_path/'frozen.csv'
    with frozen.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    result = prepare(tmp_path,tmp_path/'reused',reuse_cohort_path=frozen)
    assert read_csv(result['cohort'])[0]['body'] == 'Earlier frozen source text.'
    bindings=[json.loads(x) for x in Path(result['source_bindings']).read_text(encoding='utf-8').splitlines()]
    assert bindings[0]['kind']=='reused_frozen_cohort'
    assert 'body' in bindings[0]['fields_differing_from_current_stage1']


def test_reused_cohort_cannot_rebind_id_to_another_url(tmp_path):
    original=prepare(tmp_path,tmp_path/'first')
    rows=read_csv(original['cohort'])
    rows[0]['issue_url']='https://github.com/unrelated/repository/issues/99999'
    frozen=tmp_path/'wrong_url.csv'
    with frozen.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='reused cohort URL'):
        prepare(tmp_path,tmp_path/'reused',reuse_cohort_path=frozen)
    assert not (tmp_path/'reused'/'cohort.csv').exists()


@pytest.mark.parametrize('marker', ['not_fetched', 'not_available_in_source', 'comments_unavailable_in_source'])
def test_missing_source_status_is_not_model_evidence(marker):
    row = {key: f'  {marker.upper()}  ' for key in papers.SAFE_MODEL_FIELDS}
    assert set(papers.model_evidence_fields(row).values()) == {''}
    assert not papers._eligible('icse2023', {'issue_url': 'https://example.org/1', **row})
    substantive = f'The report literally mentions {marker} in its reproduction.'
    assert papers.model_evidence_fields({'body': substantive})['body'] == substantive


@pytest.mark.parametrize('marker', ['no_comments_in_source', '[]'])
def test_known_empty_comments_remain_distinguishable_from_unavailable(marker):
    assert papers.model_evidence_fields({'comments': marker})['comments'] == marker
    assert not papers._eligible('icse2023', {'issue_url': 'https://example.org/1', 'comments': marker})


def test_missing_source_raw_status_and_projection_version_are_auditable(tmp_path):
    paths = fixture_sources(tmp_path)
    rows = read_csv(paths[0])
    for row in rows:
        row['comments'] = 'not_fetched'
        row['state'] = 'not_available_in_source'
    write_stage(paths[0], rows)
    result = papers.prepare_paper_artifacts('icse2023', stage1_path=paths[0],
        stage2_path=paths[1], stage3_path=paths[2], output_dir=tmp_path/'out', positives=3, negatives=3)
    manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    assert manifest['evidence_projection_version'] == 'paper-primary-evidence-v2'
    bindings = [json.loads(x) for x in Path(result['source_bindings']).read_text(encoding='utf-8').splitlines()]
    snapshots = [json.loads(x) for x in Path(result['source_snapshot']).read_text(encoding='utf-8').splitlines()]
    for row, binding, snapshot in zip(read_csv(result['cohort']), bindings, snapshots):
        assert row['comments'] == row['state'] == ''
        assert binding['raw_marker_fields']['comments'] == 'not_fetched'
        assert binding['source_availability']['comments'] == 'unavailable'
        assert snapshot['stage1']['comments'] == 'not_fetched'
        assert snapshot['model_evidence']['comments'] == ''
