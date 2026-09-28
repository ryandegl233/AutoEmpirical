"""Send one tiny Gemini request; retain safe quota diagnostics, never credentials."""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmark.scripts.run_ase2022_llm_baseline import _load_env_file
from Benchmark.src.llm_provider_config import resolve_gemini_config


def summarize_response(status, body, headers, api_key):
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        data = {}
    error = data.get('error', {}) if isinstance(data, dict) else {}
    if not isinstance(error, dict):
        error = {}
    details = error.get('details', [])
    if not isinstance(details, list):
        details = []
    violations = []
    delay = None
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if 'RetryInfo' in detail.get('@type', ''):
            delay = detail.get('retryDelay')
        if 'QuotaFailure' in detail.get('@type', ''):
            for item in detail.get('violations', []):
                violations.append({k: item[k] for k in (
                    'quotaMetric', 'quotaId', 'quotaValue', 'quotaDimensions', 'description') if k in item})
    categories = set()
    message = str(error.get('message', '')).lower()
    if 'billing account has exceeded its monthly spending cap' in message:
        categories.add('monthly_spending_cap')
    for item in violations:
        text = (str(item.get('quotaMetric', '')) + str(item.get('quotaId', ''))).lower()
        if 'perday' in text or 'per_day' in text:
            categories.add('daily_tokens' if 'token' in text else 'daily_requests')
        elif 'perminute' in text or 'per_minute' in text:
            categories.add('tokens_per_minute' if 'token' in text else 'requests_per_minute')
    result = dict(http_status=status, error_status=error.get('status'),
                  message=error.get('message'), quota_violations=violations,
                  quota_categories=sorted(categories), retry_delay=delay,
                  retry_after_header=headers.get('Retry-After'),
                  cause_identified=bool(violations or categories),
                  meaning='small_request_succeeded_only' if 200 <= status < 300 else 'request_failed')
    # Never serialize raw response bytes or request headers. Redact known keys
    # and Google-key-shaped text even if the server echoes one in its message.
    serialized = json.dumps(result, ensure_ascii=True)
    if api_key:
        serialized = serialized.replace(api_key, '[REDACTED]')
    serialized = re.sub(r'AIza[0-9A-Za-z_-]{20,}', '[REDACTED]', serialized)
    return json.loads(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='gemini-3.7-flash')
    parser.add_argument('--api', choices=['openai', 'native'], default='openai')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.model):
        parser.error('Invalid model name')
    _load_env_file(ROOT / '.env')
    config = resolve_gemini_config(os.environ)
    key = config['api_key']
    if args.api == 'openai':
        url = config['base_url'] + '/chat/completions'
        headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        payload = {'model': args.model, 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                   'max_tokens': 16}
    else:
        url = 'https://generativelanguage.googleapis.com/v1beta/models/' + args.model + ':generateContent'
        headers = {'x-goog-api-key': key, 'Content-Type': 'application/json'}
        payload = {'contents': [{'parts': [{'text': 'Reply OK.'}]}],
                   'generationConfig': {'maxOutputTokens': 16}}
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
    started = time.monotonic()
    try:
        with opener.open(request, timeout=45) as response:
            result = summarize_response(response.status, response.read(262144), response.headers, key)
    except urllib.error.HTTPError as error:
        result = summarize_response(error.code, error.read(262144), error.headers, key)
    except (OSError, TimeoutError) as error:
        result = {'http_status': None, 'transport_error_type': type(error).__name__, 'cause_identified': False}
    result.update(model=args.model, api=args.api, created_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.monotonic() - started, 3), requests_sent=1)
    output = args.output or ROOT / 'reports/gemini_diagnostics' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + args.api + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=True, indent=2)
    output.write_text(text + '\n', encoding='utf-8')
    print(text)
    print('Saved:', output)
    return 0 if result.get('http_status') == 200 else 1


if __name__ == '__main__':
    raise SystemExit(main())
