"""Run frozen Gemini workflow via the explicitly recorded DragonAPI transport."""
import json
import os
import runpy
import sys
import threading
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path

from run_experiment_two_dragon import ROOT, BASE_URL, dragon_runtime, runner, should_stop
from experiment_two_resume_worker import reconcile_config


def main():
    from Benchmark.src.ase2022_llm_baseline import PooledChatCompletionClient
    from Benchmark.src.adaptive_empirical_workflow import experiment_manifest as em
    protocol_path = Path(os.environ['AE_EXPERIMENT_TWO_RESUME_PROTOCOL'])
    protocol = runner.read(protocol_path)
    runner.verify_frozen(protocol)

    def arg(name):
        return sys.argv[sys.argv.index(name) + 1]

    directory = Path(arg('--output-dir')) / arg('--experiment-id') / 'evaluation48' / arg('--arm-id') / arg('--run-id')
    if not directory.resolve().is_relative_to(protocol_path.parent.resolve()):
        raise ValueError('Worker output leaves continuation batch')
    if arg('--model') != protocol['model'] or arg('--base-url') != BASE_URL:
        raise ValueError('Model or endpoint differs from continuation protocol')
    events = protocol_path.parent / 'transport_events' / arg('--arm-id') / arg('--run-id')
    events.mkdir(parents=True, exist_ok=True)
    event_path = events / (uuid.uuid4().hex + '.jsonl')
    lock = threading.Lock()
    statuses = []
    model_mismatch = []
    original_call = PooledChatCompletionClient.call_model

    def recorded_call(self, **kwargs):
        if kwargs.get('base_url') != BASE_URL or kwargs.get('model') != protocol['model']:
            raise ValueError('Unexpected request route or model')
        event = dict(created_at=datetime.now(timezone.utc).isoformat(), provider='dragonapi',
                     base_url=BASE_URL, requested_model=kwargs['model'],
                     response_format=protocol['transport']['response_format'])
        kwargs['json_mode'] = True
        try:
            response = original_call(self, **kwargs)
        except urllib.error.HTTPError as error:
            event['http_status'] = error.code
            statuses.append(error.code)
            raise
        except Exception as error:
            event['error_type'] = type(error).__name__
            raise
        else:
            event['http_status'] = 200
            event['response_model'] = getattr(response, 'response_model', None)
            if event['response_model'] and event['response_model'] != kwargs['model']:
                model_mismatch.append(event['response_model'])
            return response
        finally:
            with lock, event_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event) + '\n')

    exit_code = 0
    with dragon_runtime(protocol):
        original_hash = em.config_hash

        def checked_hash(payload):
            if payload.get('architecture') == 'adaptive_empirical_expert_workflow':
                payload['transport'] = dict(protocol['transport'])
                payload.update(reconcile_config(payload, directory / 'run_manifest.json', protocol))
            return original_hash(payload)

        em.config_hash = checked_hash
        PooledChatCompletionClient.call_model = recorded_call
        try:
            target = ROOT / 'Benchmark/scripts/experiment_two_worker.py'
            sys.argv[0] = str(target)
            runpy.run_path(str(target), run_name='__main__')
        except SystemExit as error:
            exit_code = error.code or 0
        finally:
            em.config_hash = original_hash
            PooledChatCompletionClient.call_model = original_call
    if should_stop(statuses) or model_mismatch:
        print('Stopping batch after this case: authentication/billing/rate limit or response model mismatch. See ' + str(event_path), flush=True)
        return exit_code or 75
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
