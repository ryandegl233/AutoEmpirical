import importlib.util
import json
from pathlib import Path


def probe():
    path = Path(__file__).resolve().parents[1] / 'tools/diagnose_gemini_quota.py'
    assert path.exists(), 'quota diagnostic missing'
    spec = importlib.util.spec_from_file_location('quota_probe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_daily_quota_retry_delay_and_redacts_key():
    m = probe()
    body = json.dumps({'error': {'status': 'RESOURCE_EXHAUSTED', 'message': 'quota exceeded secret-key',
        'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [
            {'quotaMetric': 'generate_content_free_tier_requests',
             'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]},
            {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '35s'}]}}).encode()
    result = m.summarize_response(429, body, {'Retry-After': '40'}, 'secret-key')
    assert result['quota_categories'] == ['daily_requests']
    assert result['retry_delay'] == '35s'
    assert result['retry_after_header'] == '40'
    assert 'secret-key' not in json.dumps(result)
    assert result['quota_violations'][0]['quotaValue'] == '20'


def test_generic_429_does_not_invent_cause_and_success_does_not_claim_quota_available():
    m = probe()
    result = m.summarize_response(429, b'{"error":{"message":"Resource exhausted"}}', {}, 'key')
    assert result['quota_categories'] == []
    assert result['cause_identified'] is False
    assert m.summarize_response(200, b'{}', {}, 'key')['meaning'] == 'small_request_succeeded_only'
    result = m.summarize_response(429, b'non-json body with key', {}, 'key')
    assert result['cause_identified'] is False
    assert 'non-json' not in json.dumps(result)


def test_identifies_monthly_billing_cap_without_quota_details():
    m = probe()
    body = json.dumps({'error': {'status': 'RESOURCE_EXHAUSTED',
        'message': 'Your billing account has exceeded its monthly spending cap.'}}).encode()
    result = m.summarize_response(429, body, {}, 'secret')
    assert result['cause_identified'] is True
    assert result['quota_categories'] == ['monthly_spending_cap']
