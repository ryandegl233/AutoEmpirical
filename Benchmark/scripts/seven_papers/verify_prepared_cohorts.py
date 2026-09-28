"""Read-only, row-level verification of the seven prepared experiment cohorts."""
from __future__ import annotations
import hashlib
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Benchmark.src.paper_benchmark import (PAPER_IDS, SAFE_MODEL_FIELDS, build_user_prompt,
    model_evidence_fields, read_stage, canonical_url, DEFAULT_OUTPUT_ROOT,
    EVIDENCE_PROJECTION_VERSION, MISSING_SOURCE_MARKERS)
from Benchmark.scripts.prepare_paper_benchmarks import verify_preparation
from Benchmark.src.annotation_contracts import label_valid


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(base):
    results=[]; urls=defaultdict(list); checks=0
    def check(condition, message):
        nonlocal checks
        checks+=1
        if not condition:
            raise ValueError(message)
    for domain,paper_id in PAPER_IDS.items():
        folder=base/domain
        verify_preparation(folder)
        cohort=read_stage(folder/'cohort.csv',paper_id)
        sample=read_stage(folder/'stage3_sample.csv',paper_id)
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        check(manifest.get('evidence_projection_version') == EVIDENCE_PROJECTION_VERSION,
              'This verifier checks v2 only; preserve the historical v1 verification as a separate artifact.')
        taxonomy=json.loads((folder/'taxonomy.json').read_text(encoding='utf-8'))
        bindings={row['record_id']:row for row in lines(folder/'source_bindings.jsonl')}
        snapshots={row['record_id']:row for row in lines(folder/'selected_sources.jsonl')}
        counts=Counter(row['decision'] for row in cohort.values())
        check(counts=={'accepted_fault':50,'rejected_candidate':50},f'{domain}: counts')
        positives={rid for rid,row in cohort.items() if row['decision']=='accepted_fault'}
        check(set(sample)==positives,f'{domain}: Stage3 cohort differs')
        check(set(cohort)==set(bindings)==set(snapshots),f'{domain}: source IDs differ')
        check(list(cohort)==manifest['record_ids'],f'{domain}: manifest order differs')
        for rid,row in cohort.items():
            fields=model_evidence_fields(row)
            check(all(str(row.get(k,'')).strip().casefold() not in MISSING_SOURCE_MARKERS
                      for k in SAFE_MODEL_FIELDS), f'{rid}: raw missing marker in prepared evidence')
            check(bindings[rid]['evidence_projection_version'] == EVIDENCE_PROJECTION_VERSION,
                  f'{rid}: projection version missing')
            check(set(fields)==set(SAFE_MODEL_FIELDS),f'{rid}: field allowlist')
            check(fields==snapshots[rid]['model_evidence'],f'{rid}: source snapshot differs')
            check(bindings[rid]['field_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in fields.items()},f'{rid}: field hash differs')
            check(bindings[rid]['user_prompt_sha256']==hashlib.sha256(build_user_prompt(row).encode()).hexdigest(),f'{rid}: prompt hash differs')
            if rid in positives:
                check(all(label_valid(taxonomy,d,row[d]) for d in ('symptom','root_cause')),f'{rid}: invalid native gold')
            urls[canonical_url(row['issue_url'])].append({'domain':domain,'record_id':rid})
        for stage,expected in (('stage2',cohort),('stage3',sample)):
            prompts=lines(folder/f'{stage}_prompts.jsonl')
            check([p['record_id'] for p in prompts]==list(expected),f'{domain}/{stage}: prompt order')
            for prompt in prompts:
                row=cohort[prompt['record_id']]
                check(prompt['user_prompt']==build_user_prompt(row),f'{domain}/{stage}: evidence prompt altered')
                gt=row['decision'] if stage=='stage2' else {d:row[d] for d in ('symptom','root_cause')}
                check(prompt['ground_truth']==gt,f'{domain}/{stage}: gold mismatch')
        if domain=='fse2021':
            check(taxonomy['symptom']==[], 'UAV descriptions leaked into taxonomy')
        if domain=='icse2024':
            check(all(b.get('author_summary_used') is False for b in bindings.values()),'DB derived summary used')
        results.append({'domain':domain,'filter_records':len(cohort),'annotation_records':len(sample),
            'annotation_modes':taxonomy.get('annotation_modes',{}),
            'evidence_scope_counts':manifest['evidence_scope_counts']})
    overlap={url:rows for url,rows in urls.items() if len(rows)>1}
    return {'status':'PASS','evidence_projection_version':EVIDENCE_PROJECTION_VERSION,
        'row_level_checks':checks,'papers':results,
        'filter_records':sum(p['filter_records'] for p in results),
        'annotation_records':sum(p['annotation_records'] for p in results),
        'cross_paper_URL_overlap':overlap,
        'interpretation':'Paper-specific balanced cohorts; not claimed independent across papers or unseen holdout.'}


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort-root',type=Path,default=ROOT/DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'Benchmark/cache/prepared_cohort_verification_v2.json')
    args=parser.parse_args()
    report=verify(args.cohort_root)
    target=args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'papers','cross_paper_URL_overlap'}},ensure_ascii=False))
    print('Cross-paper repeated URLs:',len(report['cross_paper_URL_overlap']))
