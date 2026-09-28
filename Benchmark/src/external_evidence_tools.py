"""Read-only evidence collection, separate from registered experimental inputs.

No model invocation, label inference, OCR, or automatic trust registration.
Original responses are content-addressed; one output directory is one snapshot.
"""
from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')


def write_once(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError('refusing to replace existing snapshot: ' + str(path))


class CaptureError(Exception):
    def __init__(self, reason, status='unavailable'):
        super().__init__(reason)
        self.status = status


def validate_url(url: str, *, resolve=False, allowed_hosts=None):
    p = urlsplit(url)
    if (p.scheme not in ('http', 'https') or not p.hostname or p.username
            or p.password or p.port not in (None, 80, 443)):
        raise ValueError('only public HTTP(S) URLs on standard ports are supported')
    host = p.hostname.lower()
    if host == 'localhost' or host.endswith(('.localhost', '.local')):
        raise ValueError('local addresses are not allowed')
    if allowed_hosts is not None and host not in allowed_hosts:
        raise ValueError('host is outside configured allowlist: ' + host)
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        addresses = [ipaddress.ip_address(x[4][0]) for x in socket.getaddrinfo(host, p.port or (443 if p.scheme == 'https' else 80), type=socket.SOCK_STREAM)] if resolve else []
    if any(not address.is_global for address in addresses):
        raise ValueError('non-public network addresses are not allowed')


@dataclass
class Response:
    url: str
    status: int
    headers: dict
    body: bytes
    retrieved_at: str


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTransport:
    """Bounded GET transport. Tokens go only to HTTPS api.github.com."""
    def __init__(self, *, max_bytes=10_000_000, timeout=20, allowed_hosts=None):
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.allowed_hosts = set(allowed_hosts) if allowed_hosts else None
        self.opener = urllib.request.build_opener(NoRedirect())

    def __call__(self, url):
        deadline = time.monotonic() + self.timeout
        for _ in range(6):
            validate_url(url, resolve=True, allowed_hosts=self.allowed_hosts)
            headers = {'User-Agent': 'AutoEmpirical-Evidence/1.0', 'Accept-Encoding': 'identity'}
            if urlsplit(url).hostname == 'api.github.com' and urlsplit(url).scheme == 'https':
                headers.update({'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
                token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
                if token:
                    headers['Authorization'] = 'Bearer ' + token
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CaptureError('request_deadline_exceeded')
            try:
                response = self.opener.open(urllib.request.Request(url, headers=headers), timeout=remaining)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                if response.code in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location:
                        raise CaptureError('redirect_without_location')
                    url = urljoin(url, location)
                    continue
                chunks, size = [], 0
                while True:
                    if time.monotonic() >= deadline:
                        raise CaptureError('request_deadline_exceeded')
                    part = response.read(min(65536, self.max_bytes + 1 - size))
                    if not part:
                        break
                    chunks.append(part)
                    size += len(part)
                    if size > self.max_bytes:
                        raise CaptureError('response_exceeds_byte_limit')
                keep = {'content-type', 'link', 'last-modified', 'etag', 'retry-after', 'x-ratelimit-remaining', 'x-ratelimit-reset'}
                return Response(url, response.code, {k.lower(): v for k, v in response.headers.items() if k.lower() in keep}, b''.join(chunks), datetime.now(timezone.utc).isoformat())
        raise CaptureError('too_many_redirects')


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self.hidden = [], [], []
        self.password_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'noscript', 'svg'):
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == 'input' and attrs.get('type', '').lower() == 'password':
            self.password_form = True
        if tag in ('a', 'img'):
            value = attrs.get('href' if tag == 'a' else 'src')
            if value:
                self.links.append(value)
        if tag in ('p', 'div', 'br', 'pre', 'li', 'tr', 'h1', 'h2', 'h3'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif tag in ('p', 'div', 'pre', 'li', 'tr', 'h1', 'h2', 'h3'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class EvidenceTools:
    def __init__(self, output_dir, *, transport=None, offline=False, max_bytes=10_000_000,
                 max_chars=24000, max_pages=3, max_pixels=25_000_000):
        self.root = Path(output_dir)
        self.transport = transport or HttpTransport(max_bytes=max_bytes)
        self.offline = offline
        self.max_bytes, self.max_chars = max_bytes, max_chars
        self.max_pages, self.max_pixels = max_pages, max_pixels

    def _fetch(self, url):
        # Fragments are local locators, not HTTP request content.
        parts = urlsplit(url)
        url = urlunsplit(parts._replace(fragment=''))
        validate_url(url)
        index = self.root / 'responses' / (digest(url.encode()) + '.json')
        if index.exists():
            meta = json.loads(index.read_text(encoding='utf-8'))
            if meta['requested_url'] != url or not re.fullmatch(r'[0-9a-f]{64}', meta['raw_sha256']):
                raise CaptureError('invalid_cache_index')
            raw_path = self.root / 'raw' / meta['raw_sha256']
            raw = raw_path.read_bytes()
            if digest(raw) != meta['raw_sha256']:
                raise CaptureError('cache_hash_mismatch')
            response = Response(meta['final_url'], meta['http_status'], meta['headers'], raw, meta['retrieved_at'])
        else:
            if self.offline:
                raise CaptureError('offline_cache_miss')
            response = self.transport(url)
            validate_url(response.url)
            if len(response.body) > self.max_bytes:
                raise CaptureError('response_exceeds_byte_limit')
            sha = digest(response.body)
            meta = {'requested_url': url, 'final_url': response.url, 'http_status': response.status,
                    'headers': response.headers, 'retrieved_at': response.retrieved_at, 'raw_sha256': sha}
            write_once(self.root / 'raw' / sha, response.body)
            write_once(index, json_bytes(meta))
        if len(response.body) > self.max_bytes:
            raise CaptureError('response_exceeds_byte_limit')
        if response.status != 200:
            raise CaptureError(f'HTTP_{response.status}')
        return response

    def _json(self, url):
        r = self._fetch(url)
        try:
            return json.loads(r.body), r
        except (ValueError, UnicodeError) as error:
            raise CaptureError('invalid_json_response') from error

    def _item(self, response, content, *, kind, source_url=None, locator='', **meta):
        sha = digest(response.body)
        content_sha = digest(content.encode('utf-8'))
        identity = json_bytes([sha, source_url or response.url, locator, content_sha, meta])
        return {'evidence_id': 'E-' + digest(identity)[:24], 'kind': kind,
                'source_url': source_url or response.url, 'final_url': response.url,
                'content': content, 'content_sha256': content_sha, 'locator': locator,
                'raw_path': 'raw/' + sha, 'raw_sha256': sha,
                'retrieved_at': response.retrieved_at, 'author': None, 'published_at': None,
                'revision': None, 'temporal_status': 'unverified', **meta}

    def run(self, request):
        result = {'schema_version': 1, 'tool': request.get('tool'), 'status': 'success',
                  'items': [], 'errors': [], 'next_cursor': None}
        try:
            name = request.get('tool')
            fields = {
                'read_page': {'url', 'cursor', 'member'},
                'read_issue': {'url'},
                'read_code': {'url', 'repo', 'ref', 'path', 'start_line', 'end_line'},
                'read_image': {'url'},
            }
            if name not in fields or set(request) - fields.get(name, set()) - {'tool'}:
                raise ValueError('unknown tool or unexpected request fields')
            for key in ('cursor', 'start_line', 'end_line'):
                if key in request and (type(request[key]) is not int or request[key] < (0 if key == 'cursor' else 1)):
                    raise ValueError('invalid ' + key)
            if request.get('url'):
                validate_url(request['url'])
            if name == 'read_code' and not request.get('url'):
                if not all(request.get(x) for x in ('repo', 'ref', 'path')):
                    raise ValueError('repo, ref and path are required')
            elif not request.get('url'):
                raise ValueError('url is required')
            getattr(self, '_' + name)(request, result)
        except CaptureError as error:
            result['errors'].append({'reason': str(error)})
            result['status'] = 'partial' if result['items'] else error.status
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            # Avoid dumping credentials, response bodies, or environment variables.
            result['errors'].append({'reason': 'transport_or_storage_error:' + type(error).__name__})
            result['status'] = 'partial' if result['items'] else 'unavailable'
        except (ValueError, KeyError, TypeError) as error:
            result['errors'].append({'reason': str(error)})
            result['status'] = 'partial' if result['items'] else 'invalid_request'
        return result

    def _text(self, raw, content_type):
        if b'\x00' in raw:
            raise CaptureError('binary_content_not_supported', 'unsupported')
        charset = re.search(r'charset=["\']?([\w-]+)', content_type, re.I)
        try:
            return raw.decode(charset.group(1) if charset else 'utf-8-sig')
        except (UnicodeError, LookupError) as error:
            raise CaptureError('text_encoding_not_supported', 'unsupported') from error

    def _read_page(self, q, out):
        r = self._fetch(q['url'])
        mime = r.headers.get('content-type', '')
        raw, kind, locator = r.body, 'text', 'extracted_chars'
        metadata = {}
        if raw.startswith(b'PK\x03\x04'):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    infos = archive.infolist()
                    if len(infos) > 1000:
                        raise CaptureError('archive_entry_limit')
                    if q.get('member') is None:
                        text = '\n'.join(json.dumps({'name': x.filename, 'bytes': x.file_size}, ensure_ascii=False) for x in infos)
                        kind = 'archive_listing'
                    else:
                        info = archive.getinfo(q['member'])
                        if info.file_size > self.max_bytes:
                            raise CaptureError('archive_member_exceeds_byte_limit')
                        with archive.open(info) as handle:
                            member_raw = handle.read(self.max_bytes + 1)
                        if len(member_raw) > self.max_bytes:
                            raise CaptureError('archive_member_exceeds_byte_limit')
                        text = self._text(member_raw, '')
                        metadata['derived_from'] = digest(raw)
                        locator = 'zip_member:' + q['member'] + ':chars'
                        kind = 'archive_member'
            except (zipfile.BadZipFile, RuntimeError) as error:
                raise CaptureError('archive_unreadable') from error
        else:
            if q.get('member') is not None:
                raise ValueError('member requires a ZIP archive')
            if mime.startswith(('image/', 'audio/', 'video/')) or raw.startswith(b'%PDF'):
                raise CaptureError('use_image_tool_or_unsupported_format', 'unsupported')
            text = self._text(raw, mime)
            if 'html' in mime or re.match(r'\s*(<!doctype html|<html)', text, re.I):
                parser = PageParser()
                parser.feed(text)
                text = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', ''.join(parser.parts)).strip()
                if parser.password_form:
                    raise CaptureError('login_form_instead_of_document')
                if re.search(r'just a moment|verify you are human|enable javascript and cookies', text[:1000], re.I):
                    raise CaptureError('browser_challenge_page')
                metadata['links'] = list(dict.fromkeys(urljoin(r.url, x) for x in parser.links if urlsplit(urljoin(r.url, x)).scheme in ('http', 'https')))
                metadata['derived_from'] = digest(raw)
                kind = 'webpage'
        if not text.strip():
            raise CaptureError('empty_document')
        start = q.get('cursor', 0)
        if start >= len(text):
            raise ValueError('cursor outside document')
        end = min(start + self.max_chars, len(text))
        out['items'].append(self._item(r, text[start:end], kind=kind, source_url=q['url'],
                                      locator=f'{locator}:{start}:{end}', truncated=end < len(text), **metadata))
        if end < len(text):
            out.update(status='partial', next_cursor=end)

    def _pages(self, url, out):
        first = urlsplit(url)
        for _ in range(self.max_pages):
            try:
                data, r = self._json(url)
                if not isinstance(data, list):
                    raise CaptureError('expected_paginated_list')
                yield data, r
                next_link = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('link', ''))
                if not next_link:
                    return
                url = next_link.group(1)
                target = urlsplit(url)
                if (target.scheme, target.netloc, target.path) != (first.scheme, first.netloc, first.path):
                    raise CaptureError('invalid_pagination_origin_or_path')
            except CaptureError as error:
                out['errors'].append({'url': url, 'reason': str(error)})
                out['status'] = 'partial'
                return
        out['errors'].append({'url': url, 'reason': 'pagination_limit', 'next_url': url})
        out['status'] = 'partial'

    def _github(self, url):
        p = urlsplit(url)
        match = re.fullmatch(r'/([\w.-]+/[\w.-]+)/(issues|pull|commit)/([\w.-]+)/?', p.path)
        if p.hostname != 'github.com' or not match:
            raise ValueError('expected GitHub issue, pull, or commit URL')
        return match.groups()

    def _read_issue(self, q, out):
        repo, kind, number = self._github(q['url'])
        if kind not in ('issues', 'pull') or not number.isdigit():
            raise ValueError('expected issue or pull number')
        api = f'https://api.github.com/repos/{repo}/issues/{number}'
        data, r = self._json(api)
        if not isinstance(data, dict) or 'body' not in data:
            raise CaptureError('invalid_issue_response')

        def add(value, response, item_kind):
            content = value.get('body') or ''
            if item_kind == 'issue_body':
                content = (value.get('title') or '') + '\n\n' + content
            source = value.get('html_url') or q['url']
            out['items'].append(self._item(response, content, kind=item_kind, source_url=source,
                                          locator='body' if item_kind == 'issue_body' else 'comment:' + str(value['id']),
                                          author=(value.get('user') or {}).get('login'),
                                          published_at=value.get('created_at'), updated_at=value.get('updated_at'),
                                          author_association=value.get('author_association')))

        add(data, r, 'issue_body')
        streams = [(api + '/comments', 'issue_comment')]
        if kind == 'pull' or 'pull_request' in data:
            streams.append((f'https://api.github.com/repos/{repo}/pulls/{number}/comments', 'review_comment'))
        for endpoint, item_kind in streams:
            for values, response in self._pages(endpoint + '?per_page=100&page=1', out):
                for value in values:
                    add(value, response, item_kind)

    def _read_code(self, q, out):
        if q.get('url'):
            repo, kind, number = self._github(q['url'])
            if kind not in ('pull', 'commit'):
                raise ValueError('read_code URL must name a PR or commit')
            endpoint = f'https://api.github.com/repos/{repo}/' + ('pulls/' if kind == 'pull' else 'commits/') + number
            data, r = self._json(endpoint)
            if kind == 'pull':
                base, head = data['base']['sha'], data['head']['sha']
                if not all(re.fullmatch(r'[0-9a-f]{40}', x) for x in (base, head)):
                    raise CaptureError('invalid_pr_commit_sha')
                expected = data.get('changed_files')
                comparison_url = f'https://api.github.com/repos/{repo}/compare/{base}...{head}?per_page=1&page=1'
                comparison, diff_response = self._json(comparison_url)
                diff_base = comparison['merge_base_commit']['sha']
                if not re.fullmatch(r'[0-9a-f]{40}', diff_base):
                    raise CaptureError('invalid_diff_merge_base')
                files = comparison.get('files', [])
                pages = [(files, diff_response)]
                # GitHub returns at most 300 files on comparison page one.
                # Commit pagination does not retrieve additional changed files.
                if len(files) >= 300:
                    out['status'] = 'partial'
                    out['errors'].append({'reason': 'comparison_file_limit_300'})
            else:
                base = (data.get('parents') or [{}])[0].get('sha')
                diff_base = base
                head, expected = data['sha'], None
                # Commit responses are objects; a Link header indicates omitted files.
                pages = [(data.get('files', []), r)]
                if 'rel="next"' in r.headers.get('link', ''):
                    out['status'] = 'partial'
                    out['errors'].append({'reason': 'commit_files_paginated', 'url': endpoint})
            seen = 0
            for files, response in pages:
                for file in files:
                    patch = file.get('patch')
                    out['items'].append(self._item(response, patch or '', kind='code_diff',
                                                  source_url=q['url'], locator=file['filename'],
                                                  revision=head, base_sha=base, head_sha=head,
                                                  diff_base_sha=diff_base,
                                                  patch_available=patch is not None,
                                                  patch_completeness='not_independently_verified',
                                                  file_status=file.get('status'), file_blob_sha=file.get('sha')))
                    seen += 1
                    if patch is None:
                        out['status'] = 'partial'
                        out['errors'].append({'reason': 'patch_unavailable', 'file': file['filename']})
            if expected is not None and seen != expected:
                out['status'] = 'partial'
                out['errors'].append({'reason': 'changed_file_count_mismatch', 'expected': expected, 'received': seen})
            return
        repo, ref, path = q['repo'], q['ref'], q['path']
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo) or any(x in ('..', '') for x in path.split('/')) or '\\' in path:
            raise ValueError('invalid repo or file path')
        if q.get('end_line', q.get('start_line', 1)) < q.get('start_line', 1):
            raise ValueError('end_line precedes start_line')
        commit, _ = self._json(f'https://api.github.com/repos/{repo}/commits/{quote(ref, safe="")}')
        sha = commit['sha']
        if not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise CaptureError('invalid_resolved_commit')
        url = f'https://raw.githubusercontent.com/{repo}/{sha}/{quote(path, safe="/")}'
        r = self._fetch(url)
        lines = self._text(r.body, r.headers.get('content-type', '')).splitlines()
        start, end = q.get('start_line', 1), min(q.get('end_line', len(lines)), len(lines))
        if start > len(lines):
            raise ValueError('line range outside file')
        content = '\n'.join(lines[start-1:end])
        # Never silently crop code in the middle of a line; caller can request a range.
        if len(content) > self.max_chars:
            raise CaptureError('code_range_exceeds_text_limit_request_smaller_line_range')
        out['items'].append(self._item(r, content, kind='code', source_url=f'https://github.com/{repo}/blob/{sha}/{path}',
                                      locator=f'{path}:L{start}-L{end}', revision=sha, requested_ref=ref,
                                      published_at=commit.get('commit', {}).get('committer', {}).get('date')))

    def _read_image(self, q, out):
        from PIL import Image, UnidentifiedImageError
        r = self._fetch(q['url'])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(r.body)) as picture:
                    width, height = picture.size
                    fmt = picture.format
                    frames = getattr(picture, 'n_frames', 1)
                    if width * height > self.max_pixels:
                        raise CaptureError('image_pixel_limit')
                    if fmt not in ('PNG', 'JPEG', 'WEBP') or frames != 1:
                        raise CaptureError('image_format_or_animation_not_supported', 'unsupported')
                    picture.verify()
                # verify() checks structure but does not decode JPEG pixel data.
                with Image.open(io.BytesIO(r.body)) as picture:
                    picture.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as error:
            raise CaptureError('not_a_valid_image', 'unsupported') from error
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise CaptureError('image_pixel_limit') from error
        extension, mime = {'PNG': ('png', 'image/png'), 'JPEG': ('jpg', 'image/jpeg'), 'WEBP': ('webp', 'image/webp')}[fmt]
        asset = 'assets/' + digest(r.body) + '.' + extension
        write_once(self.root / asset, r.body)
        out['items'].append(self._item(r, '', kind='image', source_url=q['url'], locator='original_image',
                                      asset_id=digest(r.body), asset_path=asset, mime_type=mime,
                                      width=width, height=height, model_read_status='not_invoked'))

    def image_part(self, item, provider):
        """Build a real multimodal message part; this does NOT call a model."""
        if item.get('kind') != 'image' or not re.fullmatch(r'[0-9a-f]{64}', item.get('raw_sha256', '')):
            raise ValueError('expected a verified image item')
        raw = (self.root / 'raw' / item['raw_sha256']).read_bytes()
        if digest(raw) != item['raw_sha256'] or len(raw) > self.max_bytes:
            raise ValueError('image cache mismatch or byte limit')
        encoded = base64.b64encode(raw).decode('ascii')
        if provider == 'gemini':
            return {'inlineData': {'mimeType': item['mime_type'], 'data': encoded}}
        if provider == 'openai':
            return {'type': 'image_url', 'image_url': {'url': f'data:{item["mime_type"]};base64,{encoded}'}}
        raise ValueError('provider must be gemini or openai')
