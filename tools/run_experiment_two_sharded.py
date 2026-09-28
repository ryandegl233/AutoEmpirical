"""Case-isolated experiment-two entry, preserving frozen model/runtime code and prior outputs."""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Benchmark.scripts import run_experiment_two as runner

_command=runner.command
_completed_row=runner.completed_row
_batch_lock=runner.batch_lock


def command(protocol,base,arm,case=None,dry=False):
    # Each selected-record set needs its own native arm registry. Do not weaken
    # native config validation or rewrite an existing manifest to force a match.
    if case is not None and not (base/'evaluation48'/arm/f'case{case:02}'/'run_manifest.json').exists():
        base=base/'shards'/f'case{case:02}'
    return _command(protocol,base,arm,case,dry)


def completed_row(base,arm,case,rid):
    old=_completed_row(base,arm,case,rid)
    new=_completed_row(base/'shards'/f'case{case:02}',arm,case,rid)
    if old is not None and new is not None:
        raise ValueError(f'Duplicate old/sharded outputs for {arm}/case{case:02}; audit before scoring')
    return old if old is not None else new


@contextmanager
def batch_lock(base):
    with _batch_lock(base):
        audit=base/'execution_layout.json'
        expected=dict(version='case-isolated-layout-v1',entry_sha256=runner.digest(__file__),
            legacy='evaluation48/{arm}/caseNN',new='shards/caseNN/evaluation48/{arm}/caseNN',
            reason='Native arm hash includes record_ids; isolate distinct record selections.',
            model_runtime_changed=False,original_protocol_preserved=True)
        if audit.exists():
            if runner.read(audit)!=expected: raise ValueError('Execution layout changed; audit before continuing')
        else:
            audit.write_text(json.dumps(expected,indent=2),encoding='utf-8')
        yield


def main(argv=None):
    # Narrow routing adapters only. Native run/resume checks, inference, prompts,
    # concurrency, budgets, completion policy and scoring remain the frozen code.
    runner.command=command
    runner.completed_row=completed_row
    runner.batch_lock=batch_lock
    try: return runner.main(argv)
    finally:
        runner.command=_command;runner.completed_row=_completed_row;runner.batch_lock=_batch_lock


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileNotFoundError) as error: raise SystemExit(str(error))
