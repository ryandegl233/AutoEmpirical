"""Paper-specific, auditable inputs shared by the existing experiment engines.

Gold stage membership and annotations are used only for sampling and evaluation.
Model-facing content is an explicit projection of captured primary evidence.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path('Benchmark/inputs/seven_papers_v2')
EVIDENCE_PROJECTION_VERSION = 'paper-primary-evidence-v2'
DEFAULT_SEED = 20260908
ACCEPTED_FAULT = 'accepted_fault'
REJECTED_CANDIDATE = 'rejected_candidate'
SAFE_MODEL_FIELDS = ('source_project', 'issue_url', 'title', 'body', 'comments',
                     'state', 'created_at', 'changed_files', 'code_diff')
SAMPLE_FIELDS = ('record_id', 'paper_id', *SAFE_MODEL_FIELDS,
                 'decision', 'symptom', 'root_cause')
MISSING_SOURCE_MARKERS = frozenset({'not_fetched', 'not_available_in_source',
                                    'comments_unavailable_in_source'})
KNOWN_EMPTY_COMMENTS = frozenset({'no_comments_in_source', '[]'})
UNAVAILABLE = MISSING_SOURCE_MARKERS | KNOWN_EMPTY_COMMENTS | {'', 'not available'}
CODEBOOK_PATH = REPO_ROOT / 'Benchmark/configs/paper_codebooks_v1.json'
PAPER_IDS = {
    'ase2022': 'ase2022_towards_understanding_the_faults_of',
    'issta2024': 'issta2024_bugs_in_pods_understanding_bugs',
    'fse2021': 'fse2021_an_exploratory_study_of_autopilot',
    'icse2021': 'icse2021_iot_bugs_and_development_challenges',
    'icse2022': 'icse2022_an_empirical_study_on_performance',
    'icse2023': 'icse2023_an_empirical_study_on_bugs',
    'icse2024': 'icse2024_understanding_transaction_bugs_in_database',
}


@dataclass(frozen=True)
class PaperProfile:
    domain: str
    paper_id: str
    title: str
    scope: str
    filter_policy: str
    requires_repair: bool
    annotation_modes: dict[str, str]
    definitions: dict[str, dict[str, str]]
    labels: dict[str, list[str]]
    provenance: dict[str, Any]

    @property
    def symptom_mode(self) -> str:
        return self.annotation_modes.get('symptom', 'single_label')

    @property
    def default_cohort_path(self) -> str:
        return (DEFAULT_OUTPUT_ROOT / self.domain / 'cohort.csv').as_posix()

    @property
    def default_taxonomy_path(self) -> str:
        return (DEFAULT_OUTPUT_ROOT / self.domain / 'taxonomy.json').as_posix()


def _profiles() -> dict[str, PaperProfile]:
    from Benchmark.src import ase2022_llm_baseline as ase
    from Benchmark.src import issta2024_bugs_in_pods_baseline as issta
    profiles = {}
    for domain, module, title, scope in (
        ('ase2022', ase, 'Towards understanding the faults of javascript-based deep learning systems',
         'JavaScript deep-learning issue reports'),
        ('issta2024', issta, 'Bugs in Pods: Understanding Bugs in Container Runtime Systems',
         'Bug-fixing commits in container runtime systems'),
    ):
        definitions = {'symptom': dict(module.SYMPTOM_DEFINITIONS),
                       'root_cause': dict(module.ROOT_CAUSE_DEFINITIONS)}
        profiles[domain] = PaperProfile(domain, PAPER_IDS[domain], title, scope, scope,
            domain == 'issta2024', {'symptom':'single_label','root_cause':'single_label'},
            definitions, {k:list(v) for k,v in definitions.items()},
            {'status':'existing_ase_issta_adapter'})
    codebooks = json.loads(CODEBOOK_PATH.read_text(encoding='utf-8')) if CODEBOOK_PATH.exists() else {}
    for domain in PAPER_IDS.keys() - profiles.keys():
        raw = codebooks.get(domain, {})
        profiles[domain] = PaperProfile(domain, PAPER_IDS[domain], raw.get('title',domain),
            raw.get('scope',''), raw.get('filter_policy',''), bool(raw.get('requires_repair',False)),
            raw.get('annotation_modes',{}), raw.get('definitions',{}),
            raw.get('labels',{}), raw.get('provenance',{}))
    return profiles


PAPER_PROFILES = _profiles()


def get_paper_profile(domain: str) -> PaperProfile:
    try:
        profile = PAPER_PROFILES[domain]
    except KeyError as error:
        raise ValueError(f'unsupported paper domain: {domain}') from error
    if not profile.labels or not profile.definitions:
        raise ValueError(f'paper codebook unavailable for {domain}')
    return profile


paper_profile = get_paper_profile


def domain_for_record(record: dict[str, str]) -> str:
    paper_id = record.get('paper_id','') or record.get('record_id','').split(':',1)[0]
    for domain, identifier in PAPER_IDS.items():
        if paper_id == identifier:
            return domain
    raise ValueError(f'unsupported paper_id: {paper_id}')


def build_taxonomy(domain: str) -> dict[str, Any]:
    profile = get_paper_profile(domain)
    taxonomy: dict[str, Any] = copy.deepcopy(profile.labels)
    if domain not in {'ase2022','issta2024'}:
        taxonomy['annotation_modes'] = dict(profile.annotation_modes)
    return taxonomy


def model_evidence_fields(record: dict[str, str]) -> dict[str, str]:
    fields = {field: str(record.get(field,'') or '').strip() for field in SAFE_MODEL_FIELDS}
    return {field: '' if value.casefold() in MISSING_SOURCE_MARKERS else value
            for field, value in fields.items()}


def source_availability(record: dict[str, str]) -> dict[str, Any]:
    """Keep capture status in provenance, separate from the technical evidence."""
    raw = {field: str(record.get(field, '') or '').strip() for field in SAFE_MODEL_FIELDS}
    status = {}
    markers = {}
    for field, value in raw.items():
        normalized = value.casefold()
        if normalized in MISSING_SOURCE_MARKERS:
            status[field] = 'unavailable'
            markers[field] = value
        elif field == 'comments' and normalized in KNOWN_EMPTY_COMMENTS:
            status[field] = 'captured_empty'
            markers[field] = value
        else:
            status[field] = 'captured' if value else 'unspecified'
    return {'evidence_projection_version': EVIDENCE_PROJECTION_VERSION,
            'raw_marker_fields': markers, 'source_availability': status}


def build_user_prompt(record: dict[str, str]) -> str:
    fields = model_evidence_fields(record)
    return 'Classify the supplied source evidence.\n\n' + '\n\n'.join(
        f'{key}:\n{value}' for key, value in fields.items() if value)


def build_stage2_system_prompt(domain: str, *, output_contract: str | None = None) -> str:
    if domain == 'ase2022':
        from Benchmark.src.ase2022_stage2_filter_baseline import build_system_prompt
        return build_system_prompt(output_contract=output_contract)
    if domain == 'issta2024':
        from Benchmark.src.issta2024_bugs_in_pods_baseline import build_stage2_system_prompt as build
        return build(output_contract=output_contract)
    profile = get_paper_profile(domain)
    contract = output_contract or ('Return ONLY strict JSON with exactly one key decision, '
        'whose value is accepted_fault or rejected_candidate.')
    return (f'Study: {profile.title}\nResearch scope: {profile.scope}\n'
        f'{profile.filter_policy}\nUse only the supplied source evidence. '
        'Do not infer selection membership from an identifier, or request gold labels.\n' + contract)


def build_stage3_system_prompt(domain: str, taxonomy: dict[str, Any], *,
                               output_contract: str | None = None) -> str:
    if domain == 'ase2022':
        from Benchmark.src.ase2022_llm_baseline import build_system_prompt
        return build_system_prompt(taxonomy, output_contract=output_contract)
    if domain == 'issta2024':
        from Benchmark.src.issta2024_bugs_in_pods_baseline import build_stage3_system_prompt as build
        return build(taxonomy, output_contract=output_contract)
    from Benchmark.src.annotation_contracts import annotation_guidance
    profile = get_paper_profile(domain)
    expected = build_taxonomy(domain)
    if any(taxonomy.get(k) != expected.get(k) for k in ('symptom','root_cause','annotation_modes')):
        raise ValueError(f'taxonomy does not match versioned {domain} codebook')
    sections = [f'Study: {profile.title}', 'Annotate the supplied study-relevant report.',
                'A symptom describes observed behavior; a root cause describes its underlying mechanism.',
                'Codebook glosses are documented operational descriptions; they do not establish the cause of this record.']
    for dimension in ('symptom','root_cause'):
        sections.append(annotation_guidance(taxonomy, dimension))
        for label in taxonomy[dimension]:
            sections.append(f'- {label}: {profile.definitions.get(dimension,{}).get(label,label)}')
    sections.append('Use only source evidence. Distinguish supported conclusions from missing evidence. '
                    'Do not infer a code defect merely from an observed failure.')
    sections.append(output_contract or ('Return ONLY strict JSON with the string fields symptom and root_cause. '
        'For multi-label symptoms join the selected atomic labels with " || ". '
        'For free-text symptoms write a concise description grounded in the source.'))
    return '\n'.join(sections)


def build_society_task(record: dict[str, str], stage: str, taxonomy: dict[str, Any], *,
                       output_contract: str | None = None) -> str:
    domain = domain_for_record(record)
    if stage == 'stage2':
        system = build_stage2_system_prompt(domain, output_contract=output_contract)
    elif stage == 'stage3':
        system = build_stage3_system_prompt(domain,taxonomy,output_contract=output_contract)
    else:
        raise ValueError('stage must be stage2 or stage3')
    return system + '\n\n' + build_user_prompt(record)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_url(url: str) -> str:
    parsed = urlsplit(unquote(url.strip()))
    return urlunsplit((parsed.scheme.casefold(),parsed.netloc.casefold(),
                      parsed.path.rstrip('/'),parsed.query,''))


def read_stage(path: str | Path, paper_id: str | None = None) -> dict[str, dict[str,str]]:
    csv.field_size_limit(2147483647)
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'record_id','paper_id'} <= set(reader.fieldnames or []):
            raise ValueError(f'missing record_id/paper_id columns: {path}')
        records = {}
        for row in reader:
            row = {k:(v or '').strip() for k,v in row.items()}
            if paper_id and row['paper_id'] != paper_id:
                raise ValueError(f'cross-paper row in {path}: {row["paper_id"]}')
            rid = row['record_id']
            if not rid:
                raise ValueError(f'empty record_id in {path}')
            if rid in records:
                raise ValueError(f'duplicate record_id in {path}: {rid}')
            records[rid] = row
    return records


def _metadata(row: dict[str,str]) -> dict[str,Any]:
    value = json.loads(row.get('original_label_json','') or '{}')
    if not isinstance(value,dict):
        raise ValueError('original_label_json must be an object')
    return value


def _has_text(value: str) -> bool:
    return value.strip().casefold() not in UNAVAILABLE


def _database_sources(repo_root: Path) -> tuple[dict[str,dict[str,str]], list[Path]]:
    base = repo_root / 'reports/txbug_stage1_reconstruction'
    candidates = [base/'github/txbug_github_candidates.csv',
        base/'github/core_plus_tidb_label/txbug_github_candidates.csv',
        base/'mariadb/txbug_mariadb_candidates.csv',
        base/'mysql/txbug_mysql_candidates_with_text.csv',
        base/'postgresql/txbug_postgresql_candidates.csv']
    sources, used = {}, []
    for path in candidates:
        if not path.exists():
            continue
        used.append(path)
        with path.open(encoding='utf-8-sig',newline='') as handle:
            for raw in csv.DictReader(handle):
                url = canonical_url(raw.get('issue_url',''))
                if not url:
                    continue
                row = {k:(raw.get(k,'') or '').strip() for k in SAFE_MODEL_FIELDS}
                row['_source_path'] = str(path.resolve())
                old = sources.get(url)
                if old is None or len(row['body']) > len(old['body']):
                    sources[url] = row
    return sources, used


def _sampling_projection(domain: str, stage1: dict[str,str],
                         database_sources: dict[str,dict[str,str]]) -> tuple[dict[str,str],dict[str,Any]]:
    projected = {field: str(stage1.get(field, '') or '').strip() for field in SAFE_MODEL_FIELDS}
    provenance: dict[str,Any] = {'kind':'stage1_primary_source_projection',
        'record_id':stage1['record_id'], 'source_url':stage1.get('issue_url',''),
        'evidence_mode':'commit_diff' if domain == 'issta2024' else 'issue_discussion',
        'historical_snapshot':'reconstructed_source_not_asserted_as_original_snapshot'}
    if domain == 'issta2024':
        files = _metadata(stage1).get('changed_files',[])
        if not isinstance(files,list) or any(not isinstance(x,str) for x in files):
            raise ValueError(f'invalid changed_files: {stage1["record_id"]}')
        projected['changed_files'] = json.dumps(files,ensure_ascii=False)
    else:
        projected['changed_files'] = ''
        projected['code_diff'] = ''
    if domain == 'icse2024':
        raw = database_sources.get(canonical_url(stage1.get('issue_url','')))
        # The later annotated reports have author-written summaries in both fields.
        # Suppress those fields even when primary reconstruction is unavailable.
        projected['title'] = raw.get('title','') if raw else ''
        projected['body'] = raw.get('body','') if raw else ''
        if raw:
            for field in ('state','created_at'):
                if _has_text(raw.get(field,'')):
                    projected[field] = raw[field]
        provenance.update({'kind':'primary_report_and_discussion' if raw else 'primary_discussion_only',
            'primary_report_source':raw.get('_source_path') if raw else None,
            'author_summary_used':False})
    provenance.update(source_availability(projected))
    return model_evidence_fields(projected), provenance


def _eligible(domain: str, record: dict[str,str]) -> bool:
    if not _has_text(record.get('issue_url','')):
        return False
    if domain == 'ase2022' and '/pull/' in record['issue_url']:
        return False
    if domain == 'issta2024':
        diff = record.get('code_diff','')
        return 0 < len(diff.encode('utf-8')) <= 200_000 and _has_text(record.get('body',''))
    # A recovered discussion can be the only surviving primary source.
    return _has_text(record.get('body','')) or _has_text(record.get('comments',''))


def _rank(seed: int, label: str, record_id: str) -> str:
    return hashlib.sha256(f'{seed}:{label}:{record_id}'.encode()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str,str]]) -> None:
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=SAMPLE_FIELDS,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def prepare_paper_artifacts(domain: str, *, stage1_path: str | Path | None = None,
        stage2_path: str | Path | None = None, stage3_path: str | Path | None = None,
        output_dir: str | Path | None = None, positives: int = 50, negatives: int = 50,
        seed: int = DEFAULT_SEED, repo_root: str | Path = REPO_ROOT,
        reuse_cohort_path: str | Path | None = None) -> dict[str,Path]:
    from Benchmark.src.annotation_contracts import canonical_label, label_valid
    profile = get_paper_profile(domain)
    if positives <= 0 or negatives <= 0:
        raise ValueError('positives and negatives must be positive')
    root = Path(repo_root)
    base = root / 'Dataset/by_paper' / profile.paper_id
    sources = {'stage1':Path(stage1_path or base/'stage1.csv'),
               'stage2':Path(stage2_path or base/'stage2.csv'),
               'stage3':Path(stage3_path or base/'stage3.csv')}
    stages = {name:read_stage(path,profile.paper_id) for name,path in sources.items()}
    s1,s2,s3 = (stages[name] for name in ('stage1','stage2','stage3'))
    if not set(s3) <= set(s2) <= set(s1):
        raise ValueError('stage record_id lineage must satisfy Stage3 subset Stage2 subset Stage1')
    for later in (s2,s3):
        for rid,row in later.items():
            if canonical_url(row['issue_url']) != canonical_url(s1[rid]['issue_url']):
                raise ValueError(f'URL lineage mismatch: {rid}')
    taxonomy = build_taxonomy(domain)
    db_sources, extra_sources = _database_sources(root) if domain == 'icse2024' else ({},[])
    sources.update({f'primary_report_{i}':p for i,p in enumerate(extra_sources)})
    if domain not in {'ase2022','issta2024'}:
        sources['codebook'] = CODEBOOK_PATH
    selected_urls = {canonical_url(r['issue_url']) for r in s2.values()}
    examples, all_examples, provenance, exclusions = {}, {}, {}, []
    for rid,source in s1.items():
        evidence, detail = _sampling_projection(domain, source, db_sources)
        row = {'record_id':rid,'paper_id':profile.paper_id,**evidence,
               'decision':ACCEPTED_FAULT if rid in s2 else REJECTED_CANDIDATE,
               'symptom':'','root_cause':''}
        provenance[rid] = detail
        if rid in s3:
            gold = s3[rid]
            symptom = gold['symptom']
            if profile.symptom_mode == 'multi_label':
                raw_categories = _metadata(gold).get('bug_categories','')
                if not isinstance(raw_categories,str) or not raw_categories.strip():
                    raise ValueError(f'missing native multi-label gold: {rid}')
                symptom = [label.strip() for label in raw_categories.split(',') if label.strip()]
            for dimension,value in (('symptom',symptom),('root_cause',gold['root_cause'])):
                if not label_valid(taxonomy,dimension,value):
                    raise ValueError(f'gold {dimension} outside native taxonomy for {rid}: {value}')
                row[dimension] = canonical_label(taxonomy,dimension,value)
        all_examples[rid] = row
        reason = None
        if rid in s2 and rid not in s3:
            reason = 'accepted_without_final_annotation'
        elif rid not in s2 and canonical_url(row['issue_url']) in selected_urls:
            reason = 'URL_group_has_accepted_record'
        elif not _eligible(domain,row):
            reason = 'insufficient_primary_evidence_or_outside_artifact_scope'
        if reason:
            exclusions.append({'record_id':rid,'reason':reason})
        else:
            examples[rid] = row
    pools = {label:sorted((r for r in examples.values() if r['decision']==label),
                         key=lambda r:(_rank(seed,label,r['record_id']),r['record_id']))
             for label in (ACCEPTED_FAULT,REJECTED_CANDIDATE)}
    cohort = []
    used_urls: set[str] = set()
    if reuse_cohort_path is not None:
        sources['reused_cohort'] = Path(reuse_cohort_path)
        reused = read_stage(reuse_cohort_path,profile.paper_id)
        for rid,old in reused.items():
            if rid not in all_examples or (rid in s2 and rid not in s3):
                raise ValueError(f'reused cohort lacks source or final annotation: {rid}')
            row = all_examples[rid]
            if canonical_url(old.get('issue_url','')) != canonical_url(row['issue_url']):
                raise ValueError(f'reused cohort URL does not match source record identity: {rid}')
            if any(old.get(k,'') != row[k] for k in ('decision','symptom','root_cause')):
                raise ValueError(f'reused cohort gold mismatch: {rid}')
            # A frozen historical input is its own evidence version. Preserve it
            # byte-for-field instead of silently substituting current CSV text.
            old_fields = model_evidence_fields(old)
            differences = [k for k in SAFE_MODEL_FIELDS if old_fields[k] != row[k]]
            row = {**row,**old_fields}
            provenance[rid].update({'kind':'reused_frozen_cohort',
                'frozen_cohort_path':str(Path(reuse_cohort_path).resolve()),
                'fields_differing_from_current_stage1':differences,
                'meets_new_sampling_eligibility':rid in examples,
                'current_stage1_text_substituted':False})
            provenance[rid].update(source_availability(old))
            cohort.append(row)
    else:
        for label,count in ((ACCEPTED_FAULT,positives),(REJECTED_CANDIDATE,negatives)):
            chosen = []
            for row in pools[label]:
                url = canonical_url(row['issue_url'])
                if url in used_urls:
                    continue
                used_urls.add(url); chosen.append(row)
                if len(chosen)==count:
                    break
            if len(chosen) < count:
                name = 'positives' if label == ACCEPTED_FAULT else 'negatives'
                raise ValueError(f'requested {count} eligible {name}, only {len(chosen)} available')
            cohort.extend(chosen)
    cohort.sort(key=lambda r:r['record_id'])
    counts = Counter(row['decision'] for row in cohort)
    if counts != {ACCEPTED_FAULT:positives,REJECTED_CANDIDATE:negatives}:
        raise ValueError(f'cohort counts do not match request: {dict(counts)}')
    if len({canonical_url(r['issue_url']) for r in cohort}) != len(cohort):
        raise ValueError('cohort repeats a URL group')
    annotated = [r for r in cohort if r['decision']==ACCEPTED_FAULT]
    system_prompts = {'stage2':build_stage2_system_prompt(domain),
                      'stage3':build_stage3_system_prompt(domain,taxonomy)}
    output = Path(output_dir or root/DEFAULT_OUTPUT_ROOT/domain)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite existing preparation: {output}')
    output.mkdir(parents=True,exist_ok=True)
    paths = {name:output/filename for name,filename in {
        'cohort':'cohort.csv','stage3_sample':'stage3_sample.csv','taxonomy':'taxonomy.json',
        'stage2_prompts':'stage2_prompts.jsonl','stage3_prompts':'stage3_prompts.jsonl',
        'manifest':'manifest.json','source_bindings':'source_bindings.jsonl',
        'sampling_exclusions':'sampling_exclusions.jsonl','source_snapshot':'selected_sources.jsonl'}.items()}
    _write_csv(paths['cohort'],cohort); _write_csv(paths['stage3_sample'],annotated)
    paths['taxonomy'].write_text(json.dumps(taxonomy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for stage,rows in (('stage2',cohort),('stage3',annotated)):
        system = system_prompts[stage]
        with paths[f'{stage}_prompts'].open('x',encoding='utf-8') as handle:
            for row in rows:
                prompt = {'record_id':row['record_id'],'paper_id':profile.paper_id,'issue_url':row['issue_url'],
                    'ground_truth':row['decision'] if stage=='stage2' else {d:row[d] for d in ('symptom','root_cause')},
                    'system_prompt':system,'user_prompt':build_user_prompt(row)}
                handle.write(json.dumps(prompt,ensure_ascii=False)+'\n')
    with paths['source_bindings'].open('x',encoding='utf-8') as handle:
        for row in cohort:
            fields = model_evidence_fields(row)
            binding = {**provenance[row['record_id']], 'field_sha256':{
                k:hashlib.sha256(v.encode()).hexdigest() for k,v in fields.items()},
                'user_prompt_sha256':hashlib.sha256(build_user_prompt(row).encode()).hexdigest()}
            handle.write(json.dumps(binding,ensure_ascii=False)+'\n')
    with paths['source_snapshot'].open('x',encoding='utf-8') as handle:
        for row in cohort:
            rid = row['record_id']
            handle.write(json.dumps({'record_id':rid,'stage1':s1[rid],
                'stage3_gold':s3.get(rid),'model_evidence':model_evidence_fields(row)},ensure_ascii=False)+'\n')
    with paths['sampling_exclusions'].open('x',encoding='utf-8') as handle:
        for row in sorted(exclusions,key=lambda r:r['record_id']):
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest = {'schema_version':'seven-paper-preparation-v2','domain':domain,'paper_id':profile.paper_id,
        'evidence_projection_version':EVIDENCE_PROJECTION_VERSION,
        'missing_source_policy':{'exact_markers_to_empty':sorted(MISSING_SOURCE_MARKERS),
            'known_empty_comments_preserved':sorted(KNOWN_EMPTY_COMMENTS),
            'raw_status_location':'source_bindings.jsonl; original Stage1 in selected_sources.jsonl'},
        'source_paths':{k:str(p.resolve()) for k,p in sources.items()},
        'source_sha256':{k:sha256_file(p) for k,p in sources.items()},
        'source_counts':{k:len(v) for k,v in stages.items()},
        'sampling':{'seed':seed,'algorithm':'sha256_rank_url_grouped_v1','positive_count':positives,
            'negative_count':negatives,'positive_pool':'Stage3 with eligible primary evidence',
            'negative_pool':'Stage1 excluding all Stage2 URL groups',
            'eligible_pool_counts':{k:len(v) for k,v in pools.items()},
            'exclusions_by_reason':dict(Counter(x['reason'] for x in exclusions)),
            'reused_cohort':str(reuse_cohort_path) if reuse_cohort_path else None,
            'reuse_policy':'preserve_original_cohort_even_if_new_eligibility_differs' if reuse_cohort_path else None,
            'sample_is_not_claimed_as_unseen_holdout':True},
        'record_ids':[r['record_id'] for r in cohort],
        'annotation_record_ids':[r['record_id'] for r in annotated],
        'annotation_modes':profile.annotation_modes,'codebook_provenance':profile.provenance,
        'model_input_fields':list(SAFE_MODEL_FIELDS),
        'evaluation_only_fields':['decision','symptom','root_cause','original_label_json','source_file'],
        'artifact_sha256':{k:sha256_file(p) for k,p in paths.items() if k!='manifest'},
        'filter_metric_interpretation':'agreement_with_paper_selected_cohort; unselected candidates are not necessarily non-bugs',
        'evidence_scope_counts':dict(Counter(provenance[r['record_id']]['kind'] for r in cohort))}
    paths['manifest'].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return paths
