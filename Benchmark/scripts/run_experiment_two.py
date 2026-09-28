"""Four-arm rule/graph experiment. Default is OFFLINE validation; --run enables model calls."""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ARMS=('E00','E10','E01','E11')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def evidence_root():
    for parent in [ROOT,*ROOT.parents]:
        p=parent/'reports/information_final_round_20260922'
        if (p/'frozen_protocol.json').is_file(): return p
    raise ValueError('Cannot locate frozen experiment-one evidence; specify --evidence-root')


def freeze(args):
    source=args.evidence_root or evidence_root()
    previous=read(source/'frozen_protocol.json')
    ref=read(source.parent/'supplemented_information_experiment_20260922/baseline_reference.json')
    cohort=Path(ref['cohort_path'])
    if digest(cohort)!=ref['cohort_sha256']: raise ValueError('Frozen cohort changed')
    original=read(source/'runs/final-info-final48-r1-live/evaluation48/G10/r1/run_manifest.json')
    supplemental=source/'supplemental_bundle_final.json'
    if digest(supplemental)!=original['supplemental_evidence']['bundle_sha256']:
        raise ValueError('Frozen supplemental evidence changed')
    examples=ROOT/'reports/meeting_20260916/iteration_AB_v2/examples_formal_v2.json'
    example_source=ROOT/'reports/meeting_20260916/implementation_A_v1/example_source.csv'
    ids=previous['record_ids']
    if len(ids)!=48 or len(set(ids))!=48: raise ValueError('Expected the original 48 cases')
    cases=args.cases or list(range(1,49))
    if len(cases)!=len(set(cases)) or any(c<1 or c>48 for c in cases): raise ValueError('Invalid case numbers')
    if len(args.arms)!=len(set(args.arms)): raise ValueError('Duplicate arms')
    # Freeze runtime code, input files and image artifacts; never freeze credentials/.env.
    files=set((ROOT/'Benchmark/src').rglob('*.py')) | set((ROOT/'Benchmark/scripts').glob('*.py'))
    files.update([cohort,cohort.parent/'split_manifest.json',examples,example_source,supplemental])
    for row in read(supplemental)['records']:
        for image in row['images']:
            path=(supplemental.parent/image['path']).resolve()
            if digest(path)!=image['sha256']: raise ValueError('Frozen image changed: '+str(path))
            files.add(path)
    return dict(version=1,record_ids=ids,cases=cases,arms=args.arms,model=ref['model'],
        inputs=dict(cohort=str(cohort),split=str(cohort.parent/'split_manifest.json'),
            examples=str(examples),example_source=str(example_source),supplemental=str(supplemental)),
        file_hashes={str(p.resolve()):digest(p) for p in sorted(files)},
        gold_path=previous['gold_path'],gold_sha256=previous['gold_sha256'],
        graph_serialization=args.graph_serialization,
        configuration=dict(concurrency=1,case_workers=args.case_workers,max_network_retries=2,max_schema_retries=2,
            thinking_profile='label-thinking',provider='gemini',a_protocol='rules-v4',
            invalid_policy='retain_invalid_no_automatic_case_rerun'),
        design='E00 corrected baseline; E10 explicit grounding/checks; E01 source graph; E11 both. '
            'Same inputs and per-role caps. Case-wise rotated arm order. Exploratory dev evaluation.')


def verify_frozen(protocol):
    for p,h in protocol['file_hashes'].items():
        if not Path(p).is_file() or digest(p)!=h: raise ValueError('Frozen code/input changed: '+p)


def command(protocol,base,arm,case=None,dry=False):
    inp=protocol['inputs']
    run_id='offline48' if case is None else f'case{case:02}'
    cmd=[sys.executable,'-B',str(Path(__file__).with_name('experiment_two_worker.py')),
        '--domain','ase2022','--stage','stage3','--provider','gemini','--model',protocol['model'],
        '--cohort-path',inp['cohort'],'--split-manifest',inp['split'],
        '--output-dir',str(base.parent),'--experiment-id',base.name,'--arm-id',arm,'--run-id',run_id,
        '--concurrency','1','--max-network-retries','2','--max-schema-retries','2',
        '--thinking-profile','label-thinking','--module-a','--a-protocol','rules-v4','--global-supervisor',
        '--a-examples',inp['examples'],'--a-example-source',inp['example_source'],
        '--supplemental-evidence',inp['supplemental'],'--experiment-two-arm',arm,
        '--record-ids',*(protocol['record_ids'] if case is None else [protocol['record_ids'][case-1]])]
    if arm in {'E01','E11'}: cmd+=['--source-graph-mode',protocol['graph_serialization']]
    if dry: cmd+=['--dry-run']
    return cmd


def completed_row(base,arm,case,rid):
    files=list((base/'evaluation48'/arm/f'case{case:02}').glob('predictions_*.jsonl'))
    if not files: return None
    if len(files)!=1: raise ValueError('Ambiguous prediction files')
    text=files[0].read_text(encoding='utf-8').strip()
    if not text: return None
    rows=[json.loads(x) for x in text.splitlines()]
    if len(rows)!=1 or rows[0]['record_id']!=rid: raise ValueError('Corrupt or unexpected case checkpoint: '+str(files[0]))
    return rows[0]


@contextmanager
def batch_lock(base):
    """OS lock releases on exit/crash; leave the harmless lock file in place."""
    with (base/'runner.lock').open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('Batch is already running in another process: '+str(base)) from error
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def run_case_jobs(protocol,workers,job,cancel):
    """Independent cases overlap; each case retains its rotated arm sequence."""
    if workers<1: raise ValueError('case-workers must be positive')
    queue=iter(enumerate(protocol['cases']));lock=threading.Lock();stop=threading.Event()
    def worker():
        try:
            while not stop.is_set() and not cancel.is_set():
                with lock: entry=next(queue,None)
                if entry is None: return 0
                index,case=entry;arms=protocol['arms'];offset=index%len(arms)
                for arm in arms[offset:]+arms[:offset]:
                    if stop.is_set() or cancel.is_set(): return 0
                    code=job(arm,case)
                    if code:
                        stop.set();return code
            return 0
        except BaseException:
            stop.set();raise
    pool=ThreadPoolExecutor(max_workers=workers)
    try:
        futures=[pool.submit(worker) for _ in range(min(workers,len(protocol['cases'])))]
        codes=[f.result() for f in as_completed(futures)]
        return 130 if cancel.is_set() else next((c for c in codes if c),0)
    except BaseException:
        cancel.set();stop.set();raise
    finally: pool.shutdown(wait=True,cancel_futures=True)


def run_worker(cmd,env,log_path,cancel):
    """Keep each case's console separate and reap its process on interruption."""
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('ab') as log:
        with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT) as process:
            while True:
                try: return process.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    if cancel.is_set():
                        process.terminate()
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
                        return 130
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
                    raise


def score(protocol,base):
    # Called only offline after inference has completed; GT never reaches any worker.
    if digest(protocol['gold_path'])!=protocol['gold_sha256']: raise ValueError('Gold file changed')
    gold={r['record_id']:r for r in [json.loads(x) for x in Path(protocol['gold_path']).read_text(encoding='utf-8').splitlines() if x]}
    paired=[]; flags={a:[] for a in protocol['arms']}; summary={}
    for case in protocol['cases']:
        rid=protocol['record_ids'][case-1]; row=dict(case=case,record_id=rid)
        for arm in protocol['arms']:
            r=completed_row(base,arm,case,rid)
            if r is None: raise ValueError(f'Incomplete results: {case}/{arm}')
            valid=r.get('stage3_valid') is True
            joint=valid and all(r[d+'_prediction']==gold[rid][d] for d in ['symptom','root_cause'])
            row.update({arm+'_valid':valid,arm+'_symptom':r.get('symptom_prediction'),
                arm+'_root':r.get('root_cause_prediction'),arm+'_joint_correct':joint})
            flags[arm].append(joint)
        paired.append(row)
    for arm in protocol['arms']:
        summary[arm]=dict(n=len(paired),valid=sum(r[arm+'_valid'] for r in paired),
            joint_correct=sum(flags[arm]),joint_accuracy=sum(flags[arm])/len(paired),
            invalid_cases=[r['case'] for r in paired if not r[arm+'_valid']])
        summary[arm]['unknown_count']=sum(r[arm+'_valid'] and r[arm+'_root']=='Unknown' for r in paired)
        events=[read(p) for p in (base/'request_traces'/arm).rglob('*.response.json')]
        summary[arm]['cost']=dict(transport_attempts=len(events),
            failed_transport_attempts=sum(e['status']=='transport_failed' for e in events),
            reported_prompt_tokens=sum(e.get('prompt_tokens') or 0 for e in events),
            reported_completion_tokens=sum(e.get('completion_tokens') or 0 for e in events),
            missing_usage_count=sum(e.get('prompt_tokens') is None or e.get('completion_tokens') is None for e in events),
            summed_call_seconds=sum(e.get('elapsed_seconds',0) for e in events))
        for d,key in [('symptom','symptom'),('root_cause','root')]:
            summary[arm][d+'_correct']=sum(r[arm+'_valid'] and r[arm+'_'+key]==gold[r['record_id']][d] for r in paired)
    comparisons={}
    from scipy.stats import binomtest
    import numpy as np
    for a,b in [('E00','E10'),('E00','E01'),('E10','E11'),('E01','E11')]:
        if a not in flags or b not in flags: continue
        gain=sum(not x and y for x,y in zip(flags[a],flags[b])); loss=sum(x and not y for x,y in zip(flags[a],flags[b]))
        comparisons[b+'-'+a]=dict(gain=gain,loss=loss,delta=(gain-loss)/len(paired),
            exact_mcnemar_p=binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.,
            note='Exploratory, unadjusted multiple comparisons; invalid outputs counted wrong, not assumed unknowable labels are wrong.')
        differences=np.asarray(flags[b],dtype=int)-np.asarray(flags[a],dtype=int)
        rng=np.random.default_rng(20260922)
        bootstrap=rng.choice(differences,size=(10000,len(paired)),replace=True).mean(axis=1)
        comparisons[b+'-'+a]['paired_bootstrap_95_interval']=np.quantile(bootstrap,[.025,.975]).tolist()
        comparisons[b+'-'+a]['interval_note']='Percentile paired-case bootstrap, seed 20260922; conditional on these dev cases and one run, not model-run variability.'
        comparisons[b+'-'+a]['meets_exploratory_5pp_threshold']=(gain-loss)/len(paired)>=.05
        delta=(gain-loss)/len(paired)
        comparisons[b+'-'+a]['invalid_label_sensitivity_bounds']=[
            delta-len(summary[a]['invalid_cases'])/len(paired),
            delta+len(summary[b]['invalid_cases'])/len(paired)]
    output=base/'analysis';output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(dict(arms=summary,comparisons=comparisons),ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'paired_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-id',required=True)
    p.add_argument('--run',action='store_true',help='Enable paid model calls; omit for offline validation.')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--score-only',action='store_true')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--cases',nargs='+',type=int)
    p.add_argument('--graph-serialization',choices=['graph','flat'],default='graph')
    p.add_argument('--evidence-root',type=Path)
    p.add_argument('--case-workers',type=int,help='Concurrent independent cases (default 1; resume inherits frozen value).')
    args=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.batch_id): p.error('Invalid batch ID')
    if args.case_workers is not None and args.case_workers<1: p.error('case-workers must be positive')
    base=ROOT/'reports/experiment_two'/(args.batch_id if args.run or args.score_only else args.batch_id+'-check')
    frozen=base/'protocol.json'
    if args.resume or args.score_only:
        if not frozen.exists(): p.error('No frozen batch to resume')
        protocol=read(frozen);verify_frozen(protocol)
        workers=protocol['configuration'].get('case_workers',1)
        if args.case_workers is not None and args.case_workers!=workers: p.error('case-workers differs from frozen batch')
        args.case_workers=workers
        if not args.score_only and (args.arms!=protocol['arms'] or args.graph_serialization!=protocol['graph_serialization']
            or args.cases is not None and args.cases!=protocol['cases']): p.error('Resume options differ from frozen batch')
    else:
        if base.exists(): p.error('Batch already exists; use --resume or another batch ID')
        args.case_workers=args.case_workers or 1
        protocol=freeze(args);base.mkdir(parents=True)
        frozen.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    cancel=threading.Event()
    def job(arm,case):
        verify_frozen(protocol)
        if case is not None and completed_row(base,arm,case,protocol['record_ids'][case-1]) is not None:
            print(f'Skip completed {case:02}/{arm}',flush=True);return 0
        print(f'{arm} case={case or "all48"} mode={"LIVE" if args.run else "OFFLINE"}',flush=True)
        env=os.environ.copy();env['AE_EXPERIMENT_TWO_TRACE_DIR']=str(base/'request_traces'/arm/(f'case{case:02}' if case else 'offline'))
        log=base/'worker_logs'/arm/(f'case{case:02}.log' if case else 'offline.log')
        code=run_worker(command(protocol,base,arm,case,dry=not args.run),env,log,cancel)
        print(f'{arm} case={case or "all48"} exit={code} log={log}',flush=True)
        return code
    with batch_lock(base):
        if args.score_only: score(protocol,base);return 0
        if args.run:
            code=run_case_jobs(protocol,args.case_workers,job,cancel)
            if code: return code
            score(protocol,base)
        else:
            for arm in protocol['arms']:
                code=job(arm,None)
                if code: return code
    print('Results: '+str(base),flush=True)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileNotFoundError) as error: raise SystemExit(str(error))
