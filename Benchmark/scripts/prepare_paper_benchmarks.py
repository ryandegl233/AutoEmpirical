"""Prepare or verify all seven paper cohorts without modifying source data."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

from Benchmark.src.paper_benchmark import (DEFAULT_OUTPUT_ROOT, DEFAULT_SEED,
    PAPER_IDS, prepare_paper_artifacts, sha256_file)


def verify_preparation(directory: Path, *, check_sources: bool = True,
                       repository_root: Path | None = None) -> dict:
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    relocation_path = directory/'release_paths.json'
    relocation = None
    root = (repository_root or REPO_ROOT).resolve()
    if relocation_path.exists():
        relocation = json.loads(relocation_path.read_text(encoding='utf-8'))
        if relocation.get('original_manifest_sha256') != sha256_file(manifest_path):
            raise ValueError('release binding manifest hash mismatch')
        if (relocation.get('schema_version') != 1
                or set(relocation.get('sources', {})) != set(manifest['source_sha256'])):
            raise ValueError('release source binding schema mismatch')
    filenames = {'cohort':'cohort.csv','taxonomy':'taxonomy.json','stage3_sample':'stage3_sample.csv',
        'stage2_prompts':'stage2_prompts.jsonl','stage3_prompts':'stage3_prompts.jsonl',
        'source_bindings':'source_bindings.jsonl','sampling_exclusions':'sampling_exclusions.jsonl',
        'source_snapshot':'selected_sources.jsonl'}
    errors = []
    for name,digest in manifest['artifact_sha256'].items():
        path = directory/filenames[name]
        if not path.exists() or sha256_file(path) != digest:
            errors.append(f'artifact hash mismatch: {name}')
    if check_sources:
        for name,digest in manifest['source_sha256'].items():
            binding = relocation['sources'][name] if relocation else None
            path = (root / binding['path']).resolve() if binding else Path(manifest['source_paths'][name])
            if binding and (not path.is_relative_to(root) or Path(binding['path']).is_absolute()):
                raise ValueError('release source path escapes repository')
            if binding and binding['sha256'] != digest:
                raise ValueError('release source binding differs from original digest')
            raw = path.read_bytes() if path.is_file() else None
            exact = raw is not None and hashlib.sha256(raw).hexdigest() == digest
            # Git may materialize existing source CSVs with LF instead of CRLF.
            # Frozen input artifacts above still require exact byte hashes.
            normalized = (binding and raw is not None and
                hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == binding['normalized_lf_sha256'])
            if not exact and not normalized:
                errors.append(f'source hash mismatch: {name}')
    if errors:
        raise ValueError('; '.join(errors))
    return {'status':'PASS','domain':manifest['domain'],
            'records':len(manifest['record_ids']),
            'annotation_records':len(manifest['annotation_record_ids'])}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains',nargs='+',choices=tuple(PAPER_IDS),default=list(PAPER_IDS))
    parser.add_argument('--output-root',default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--positives',type=int,default=50)
    parser.add_argument('--negatives',type=int,default=50)
    parser.add_argument('--verify-existing',action='store_true')
    parser.add_argument('--reuse-ase-cohort',default=None,
        help='Preserve frozen ASE input text exactly; validate IDs/gold and record differences from current Stage1.')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    results=[]
    for domain in args.domains:
        output=Path(args.output_root)/domain
        if args.verify_existing:
            results.append(verify_preparation(output))
        else:
            paths=prepare_paper_artifacts(domain,output_dir=output,seed=args.seed,
                positives=args.positives,negatives=args.negatives,
                reuse_cohort_path=args.reuse_ase_cohort if domain=='ase2022' else None)
            results.append({'domain':domain,**{k:str(v.resolve()) for k,v in paths.items()}})
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
