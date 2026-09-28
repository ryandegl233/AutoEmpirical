"""Collect public evidence from explicit requests; never runs an LLM."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Benchmark.src.external_evidence_tools import EvidenceTools, HttpTransport, digest, json_bytes, write_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--offline', action='store_true', help='Only replay captured responses')
    parser.add_argument('--allow-host', action='append', help='Exact host allowlist, including redirects')
    args = parser.parse_args(argv)
    try:
        raw = args.requests.read_bytes()
        rows = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(rows, list) or not rows or len(rows) > 384:
            raise ValueError('requests must contain 1..384 records')
        ids, counts = set(), Counter()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'request_id', 'case_id', 'request'}:
                raise ValueError('manifest records allow only request_id, case_id, request')
            if not isinstance(row['request_id'], str) or not row['request_id'] or row['request_id'] in ids:
                raise ValueError('request_id must be unique nonempty text')
            if not isinstance(row['case_id'], str) or not isinstance(row['request'], dict):
                raise ValueError('case_id must be text and request must be an object')
            ids.add(row['request_id'])
            counts[row['case_id']] += 1
        if any(n > 8 for n in counts.values()):
            raise ValueError('initial collection budget: at most 8 tool calls per case per manifest')
        if args.results.exists():
            raise ValueError('results already exist; choose a new path to preserve previous results')
        transport = HttpTransport(allowed_hosts=args.allow_host)
        tools = EvidenceTools(args.output_dir, transport=transport, offline=args.offline)
        results = []
        for row in rows:
            result = tools.run(row['request'])
            result_sha = digest(json_bytes(result))
            write_once(args.output_dir / 'results' / (result_sha + '.json'), json_bytes(result))
            results.append({**row, 'result': result, 'result_sha256': result_sha})
            print(json.dumps({'request_id': row['request_id'], 'status': result['status'],
                              'items':len(result['items']), 'errors':result['errors']}, ensure_ascii=False), flush=True)
        summary = {'schema_version':1, 'requests_sha256':digest(raw),
                   'created_at':datetime.now(timezone.utc).isoformat(), 'offline':args.offline,
                   'model_invoked':False, 'status_counts':dict(Counter(x['result']['status'] for x in results)),
                   'records':results}
        write_once(args.results, json_bytes(summary))
        return 0 if all(x['result']['status'] == 'success' for x in results) else 1
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
