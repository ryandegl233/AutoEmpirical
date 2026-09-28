"""Apply transport safeguards without changing frozen inference or old manifests."""
import os
import sys
from pathlib import Path

import run_experiment_two_dragon as legacy
from run_experiment_two_dragon_safe import safe_transport_manifest
from dragon_transport_guard import guarded_client
import experiment_two_dragon_worker as worker


def main():
    base = Path(os.environ['AE_EXPERIMENT_TWO_RESUME_PROTOCOL']).parent
    arm = sys.argv[sys.argv.index('--arm-id') + 1]
    case = sys.argv[sys.argv.index('--run-id') + 1]
    old_manifest = legacy.transport_manifest
    old_stop = worker.should_stop
    legacy.transport_manifest = safe_transport_manifest
    try:
        with guarded_client(base / 'transport_diagnostics' / arm / case) as guard:
            worker.should_stop = lambda statuses: guard.blocked or old_stop(statuses)
            result = worker.main()
            if guard.blocked:
                print('Provider unavailable; batch stopped: ' + str(guard.block_reason), flush=True)
            return result
    finally:
        legacy.transport_manifest = old_manifest
        worker.should_stop = old_stop


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
