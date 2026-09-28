"""UTF-8 entry; --resume skips valid and retries invalid/missing case/arm runs."""
import os
import sys
from pathlib import Path


def main():
    # PYTHONIOENCODING is read at interpreter startup: reconfigure this process
    # as well as setting the inherited environment for every model worker.
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONUNBUFFERED']='1'
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from run_experiment_two_sharded import main as run
    from Benchmark.scripts import run_experiment_two as runner
    from experiment_two_resume_policy import resume_policy
    with resume_policy(runner, sys.argv[1:]):
        if '--serial' in sys.argv[1:]:
            return run([v for v in sys.argv[1:] if v != '--serial'])
        return run()


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileNotFoundError) as error: raise SystemExit(str(error))
