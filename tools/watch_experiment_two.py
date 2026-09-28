"""Read-only progress display. Does not import or change experimental runtime code."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def snapshot(base):
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    result=dict(total=len(protocol['cases'])*len(protocol['arms']),done=0,valid=0,
                invalid=0,arms={},started=[],unreadable=[])
    for arm in protocol['arms']:
        count=0
        for case in protocol['cases']:
            key=f'{arm}/case{case:02}'
            directory=base/'evaluation48'/arm/f'case{case:02}'
            directories=[directory,base/'shards'/f'case{case:02}'/'evaluation48'/arm/f'case{case:02}']
            paths=[p for d in directories for p in d.glob('predictions_*.jsonl')]
            if paths:
                raw=None
                try:
                    if len(paths)!=1: raise ValueError('Ambiguous result')
                    raw=paths[0].read_text(encoding='utf-8')
                    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
                    if len(rows)!=1 or rows[0].get('record_id')!=protocol['record_ids'][case-1]:
                        raise ValueError('Incomplete or mismatched result')
                    if type(rows[0].get('stage3_valid')) is not bool: raise ValueError('Incomplete result')
                except (ValueError,OSError):
                    if raw is None or raw.strip(): result['unreadable'].append(key)
                else:
                    result['done']+=1;count+=1
                    result['valid' if rows[0]['stage3_valid'] else 'invalid']+=1
                    continue
            trace=base/'request_traces'/arm/f'case{case:02}'
            requests=list(trace.glob('*.request.json'));responses=list(trace.glob('*.response.json'))
            log=base/'worker_logs'/arm/f'case{case:02}.log'
            if requests or responses or any(d.exists() for d in directories) or log.exists():
                files=[*requests,*responses,*(p for d in directories for p in d.glob('run_manifest.json'))]
                if log.exists(): files.append(log)
                times=[]
                for p in files:
                    try: times.append(p.stat().st_mtime)
                    except OSError: pass
                result['started'].append(dict(case=key,requests=len(requests),responses=len(responses),
                    last_activity=max(times) if times else None))
        result['arms'][arm]=count
    return result


def render(state,batch):
    total=state['total'];done=state['done'];fraction=done/total if total else 0
    filled=int(fraction*30)
    lines=[f'Batch: {batch}',f'[{"#"*filled}{"-"*(30-filled)}] {done}/{total} ({fraction:.1%})',
        f'Valid: {state["valid"]} | Invalid: {state["invalid"]}',
        ' | '.join(f'{a}: {n}/{total//max(len(state["arms"]),1)}' for a,n in state['arms'].items()),'']
    for item in state['started']:
        age=f'{max(0,int(time.time()-item["last_activity"]))}s' if item['last_activity'] else '?'
        lines.append(f'{item["case"]}: attempts started/ended {item["requests"]}/{item["responses"]}; last activity {age} ago')
    if state['unreadable']: lines.append('Partial/unreadable result: '+', '.join(state['unreadable']))
    lines+=['','Started without result may be running OR interrupted. This display does not prove process liveness.',
            'Progress counts case-arm outputs, including invalid ones. Ctrl+C closes this monitor only.']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-id',default='exp2-parallel-v1')
    parser.add_argument('--interval',type=float,default=2)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<1: parser.error('interval must be at least 1 second')
    base=ROOT/'reports/experiment_two'/args.batch_id
    if not (base/'protocol.json').is_file(): parser.error('Batch protocol not found: '+str(base))
    try:
        while True:
            state=snapshot(base)
            if sys.stdout.isatty() and not args.once: print('\033[2J\033[H',end='')
            print(render(state,args.batch_id),flush=True)
            if args.once or state['done']==state['total']: return 0
            time.sleep(args.interval)
    except KeyboardInterrupt: return 0


if __name__=='__main__': raise SystemExit(main())
