import base64
import hashlib
import importlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    m = importlib.import_module('Benchmark.src.external_evidence_tools')
except ModuleNotFoundError:
    m = None


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(m, 'external evidence tools have not been implemented')
        # Retain test artifacts: workspace policy forbids recursive deletion.
        self.root = Path(tempfile.mkdtemp(prefix='evidence-tools-test-'))
        self.responses = {}
        self.calls = []

        def transport(url):
            self.calls.append(url)
            if url not in self.responses:
                raise AssertionError('unexpected URL: ' + url)
            value = self.responses[url]
            if isinstance(value, Exception):
                raise value
            return value

        self.tools = m.EvidenceTools(self.root, transport=transport)

    def response(self, url, body, mime='text/plain', status=200, **headers):
        if isinstance(body, (list, dict)):
            body = json.dumps(body).encode()
            mime = 'application/json'
        elif isinstance(body, str):
            body = body.encode()
        self.responses[url] = m.Response(url, status, {'content-type': mime, **headers}, body, '2026-09-22T00:00:00+00:00')

    def test_html_preserves_source_and_excludes_script(self):
        url = 'https://example.org/doc'
        self.response(url, '<html><script>bad()</script><h1>API</h1><pre>x = 1\ny = 2</pre><a href="/ref">reference</a></html>', 'text/html')
        r = self.tools.run({'tool': 'read_page', 'url': url})
        self.assertEqual(r['status'], 'success')
        item = r['items'][0]
        self.assertNotIn('bad()', item['content'])
        self.assertIn('x = 1\ny = 2', item['content'])
        self.assertEqual(item['links'], ['https://example.org/ref'])
        raw = (self.root / item['raw_path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), item['raw_sha256'])
        self.assertEqual(item['temporal_status'], 'unverified')

    def test_http_failure_and_login_not_empty_evidence(self):
        for status in (403, 404, 429):
            url = f'https://example.org/{status}'
            self.response(url, 'failed', status=status)
            r = self.tools.run({'tool': 'read_page', 'url': url})
            self.assertEqual(r['status'], 'unavailable')
            self.assertIn(str(status), r['errors'][0]['reason'])
        url = 'https://example.org/private'
        self.response(url, '<html><form><input type="password">Sign in</form></html>', 'text/html')
        self.assertEqual(self.tools.run({'tool': 'read_page', 'url': url})['status'], 'unavailable')

    def test_offline_replay_and_tampered_cache(self):
        url = 'https://example.org/log'
        self.response(url, 'failure at line 12')
        r = self.tools.run({'tool': 'read_page', 'url': url})
        offline = m.EvidenceTools(self.root, offline=True)
        self.assertEqual(offline.run({'tool': 'read_page', 'url': url})['items'], r['items'])
        self.assertEqual(len(self.calls), 1)
        (self.root / r['items'][0]['raw_path']).write_bytes(b'changed')
        self.assertEqual(offline.run({'tool': 'read_page', 'url': url})['status'], 'unavailable')
        self.assertEqual(offline.run({'tool': 'read_page', 'url': url + 'missing'})['status'], 'unavailable')

    def test_text_pagination_and_binary_rejection(self):
        url = 'https://example.org/log'
        self.response(url, '0123456789')
        self.tools.max_chars = 4
        r = self.tools.run({'tool': 'read_page', 'url': url})
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(r['items'][0]['content'], '0123')
        self.assertEqual(r['next_cursor'], 4)
        r = self.tools.run({'tool': 'read_page', 'url': url, 'cursor': 8})
        self.assertEqual(r['items'][0]['content'], '89')
        self.response(url + '.bin', b'\x00\x01', 'application/octet-stream')
        self.assertEqual(self.tools.run({'tool': 'read_page', 'url': url + '.bin'})['status'], 'unsupported')

    def test_issue_pagination_retains_author_and_failure(self):
        api = 'https://api.github.com/repos/a/b/issues/1'
        self.response(api, {'body': 'observed failure', 'title': 'Bug', 'html_url': 'https://github.com/a/b/issues/1', 'user': {'login': 'alice'}, 'created_at': '2020-01-01', 'updated_at': '2020-01-02'})
        page = api + '/comments?per_page=100&page=1'
        page2 = api + '/comments?per_page=100&page=2'
        self.response(page, [{'id': 123, 'body': 'my hypothesis', 'user': {'login': 'bob'}, 'created_at': '2020-01-03', 'html_url': 'https://github.com/a/b/issues/1#issuecomment-123'}], link=f'<{page2}>; rel="next"')
        self.response(page2, 'rate limited', status=403)
        r = self.tools.run({'tool': 'read_issue', 'url': 'https://github.com/a/b/issues/1'})
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(len(r['items']), 2)
        self.assertEqual(r['items'][1]['author'], 'bob')
        self.assertEqual(r['items'][1]['locator'], 'comment:123')
        self.assertIn('403', r['errors'][0]['reason'])

    def test_code_resolves_ref_and_reads_immutable_path(self):
        sha = 'a' * 40
        self.response('https://api.github.com/repos/a/b/commits/v1', {'sha': sha, 'commit': {'committer': {'date': '2020-01-01'}}})
        self.response(f'https://raw.githubusercontent.com/a/b/{sha}/src/a.py', 'one\ntwo\nthree\n')
        r = self.tools.run({'tool': 'read_code', 'repo': 'a/b', 'ref': 'v1', 'path': 'src/a.py', 'start_line': 2, 'end_line': 3})
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['items'][0]['content'], 'two\nthree')
        self.assertEqual(r['items'][0]['revision'], sha)
        self.assertEqual(r['items'][0]['locator'], 'src/a.py:L2-L3')
        self.assertNotIn('/v1/src/a.py', self.calls[-1])

    def test_pr_without_patch_is_partial(self):
        api = 'https://api.github.com/repos/a/b/pulls/2'
        self.response(api, {'base': {'sha': 'a'*40}, 'head': {'sha': 'b'*40}, 'changed_files': 1, 'updated_at': '2020-01-01'})
        compare = 'https://api.github.com/repos/a/b/compare/' + 'a'*40 + '...' + 'b'*40 + '?per_page=1&page=1'
        self.response(compare, {'merge_base_commit': {'sha': 'a'*40}, 'files': [{'filename': 'blob.png', 'status': 'modified', 'sha': 'c'*40}]})
        r = self.tools.run({'tool': 'read_code', 'url': 'https://github.com/a/b/pull/2'})
        self.assertEqual(r['status'], 'partial')
        self.assertFalse(r['items'][0]['patch_available'])
        self.assertEqual(r['items'][0]['base_sha'], 'a'*40)

    def test_pr_diff_is_fetched_by_captured_shas(self):
        api = 'https://api.github.com/repos/a/b/pulls/3'
        self.response(api, {'base': {'sha': 'a'*40}, 'head': {'sha': 'b'*40}, 'changed_files': 1})
        compare = 'https://api.github.com/repos/a/b/compare/' + 'a'*40 + '...' + 'b'*40 + '?per_page=1&page=1'
        self.response(compare, {'merge_base_commit': {'sha': 'd'*40}, 'files': [{'filename': 'a.py', 'patch': '@@ -1 +1 @@\n-old\n+new'}]})
        r = self.tools.run({'tool': 'read_code', 'url': 'https://github.com/a/b/pull/3'})
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['items'][0]['diff_base_sha'], 'd'*40)
        self.assertIn(compare, self.calls)
        self.assertFalse(any('/pulls/3/files' in c for c in self.calls))

    def test_image_returns_real_bytes_for_both_protocols(self):
        from PIL import Image
        out = io.BytesIO()
        Image.new('RGB', (12, 8), 'red').save(out, format='PNG')
        raw = out.getvalue()
        url = 'https://example.org/screenshot.png'
        self.response(url, raw, 'image/png')
        r = self.tools.run({'tool': 'read_image', 'url': url})
        self.assertEqual(r['status'], 'success')
        item = r['items'][0]
        self.assertEqual(item['width'], 12)
        gemini = self.tools.image_part(item, 'gemini')
        self.assertEqual(base64.b64decode(gemini['inlineData']['data']), raw)
        openai = self.tools.image_part(item, 'openai')
        self.assertEqual(base64.b64decode(openai['image_url']['url'].split(',')[1]), raw)
        self.assertNotIn('ocr', item)

    def test_fake_and_oversize_images_rejected(self):
        from PIL import Image
        url = 'https://example.org/fake.png'
        self.response(url, '<html>oops</html>', 'image/png')
        self.assertEqual(self.tools.run({'tool': 'read_image', 'url': url})['status'], 'unsupported')
        out = io.BytesIO()
        Image.new('RGB', (50, 50)).save(out, format='PNG')
        self.response(url+'2', out.getvalue(), 'image/png')
        self.tools.max_pixels = 100
        self.assertEqual(self.tools.run({'tool': 'read_image', 'url': url+'2'})['status'], 'unavailable')

    def test_truncated_jpeg_is_not_accepted(self):
        from PIL import Image
        out = io.BytesIO()
        Image.new('RGB', (100, 100), 'red').save(out, format='JPEG')
        url = 'https://example.org/truncated.jpg'
        self.response(url, out.getvalue()[:-20], 'image/jpeg')
        r = self.tools.run({'tool':'read_image', 'url':url})
        self.assertEqual(r['status'], 'unsupported')
        self.assertEqual(r['items'], [])

    def test_zip_listing_selected_text_and_size_limit(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as archive:
            archive.writestr('../model.json', '{"model": 1}')
            archive.writestr('large.log', 'x'*1000)
        url = 'https://example.org/model.zip'
        self.response(url, out.getvalue(), 'application/zip')
        r = self.tools.run({'tool': 'read_page', 'url': url})
        self.assertEqual(r['items'][0]['kind'], 'archive_listing')
        self.assertFalse((self.root.parent/'model.json').exists())
        r = self.tools.run({'tool': 'read_page', 'url': url, 'member': '../model.json'})
        self.assertIn('model', r['items'][0]['content'])
        self.tools.max_bytes = 200
        r = self.tools.run({'tool': 'read_page', 'url': url, 'member': 'large.log'})
        self.assertEqual(r['status'], 'unavailable')

    def test_bad_request_fields_and_urls(self):
        for request in [
            {'tool':'read_page', 'url':'file:///C:/secret'},
            {'tool':'read_page', 'url':'http://127.0.0.1/a'},
            {'tool':'read_page', 'url':'https://example.org/a', 'gt':'x'},
            {'tool':'read_page', 'url':'https://example.org/a', 'cursor':-1},
            {'tool':'read_code', 'repo':'a/b', 'path':'x'},
        ]:
            self.assertEqual(self.tools.run(request)['status'], 'invalid_request')
        self.assertEqual(self.calls, [])

    def test_cli_offline_writes_results_and_rejects_audit_fields(self):
        import importlib.util
        spec = importlib.util.find_spec('Benchmark.scripts.collect_external_evidence')
        self.assertIsNotNone(spec, 'collection CLI has not been implemented')
        cli = importlib.import_module('Benchmark.scripts.collect_external_evidence')
        url = 'https://example.org/log'
        self.response(url, 'log content')
        self.tools.run({'tool':'read_page', 'url':url})
        manifest = self.root / 'requests.json'
        manifest.write_text(json.dumps([{'request_id':'case-log', 'case_id':'7', 'request':{'tool':'read_page', 'url':url}}]), encoding='utf-8')
        report = self.root / 'results.json'
        self.assertEqual(cli.main(['--requests', str(manifest), '--output-dir', str(self.root), '--results', str(report), '--offline']), 0)
        saved = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(saved['records'][0]['result']['status'], 'success')
        self.assertFalse(saved['model_invoked'])
        manifest.write_text(json.dumps([{'request_id':'a', 'case_id':'7', 'gt':'answer', 'request':{'tool':'read_page', 'url':url}}]), encoding='utf-8')
        self.assertEqual(cli.main(['--requests', str(manifest), '--output-dir', str(self.root), '--results', str(self.root/'other.json'), '--offline']), 2)


if __name__ == '__main__':
    unittest.main()
