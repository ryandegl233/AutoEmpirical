"""Frozen supplemental facts and original pixels for the information-only arm.

No retrieval, caption model, label inference, or modification of role instructions.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from pathlib import Path

from PIL import Image

from .contracts import EvidenceItem


_METADATA_KEYS = {
    'supplemental_image_sha256', 'source_author', 'source_published_at',
    'source_revision', 'source_raw_sha256', 'source_locator', 'reference_scope',
    'collection_request_id', 'content_scope',
}
_SOURCE_TYPES = {'issue_body', 'issue_comment', 'source_code', 'code_diff',
                 'execution_log', 'runtime_observation', 'record_summary'}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class SupplementalEvidenceBundle:
    def __init__(self):
        self._items = {}
        self._images = {}
        self._events = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path, *, allowed_record_ids):
        path = Path(path).resolve()
        raw = path.read_bytes()
        payload = json.loads(raw.decode('utf-8-sig'))
        if set(payload) != {'schema_version', 'information_policy', 'records'}:
            raise ValueError('supplement bundle has unexpected fields')
        if payload['schema_version'] != 1 or payload['information_policy'] != 'public_at_collection':
            raise ValueError('unsupported supplemental evidence policy')
        if not isinstance(payload['records'], list) or not payload['records']:
            raise ValueError('supplement bundle must contain records')
        bundle = cls()
        bundle.sha256 = _sha(raw)
        seen = set()
        for row in payload['records']:
            if set(row) != {'record_id', 'items', 'images'}:
                raise ValueError('supplement record has unexpected fields')
            rid = row['record_id']
            if rid not in allowed_record_ids or rid in bundle._items:
                raise ValueError('unknown or duplicate supplement record_id')
            if not isinstance(row['items'], list) or not 1 <= len(row['items']) <= 80:
                raise ValueError('supplement record requires 1..80 evidence items')
            items = tuple(EvidenceItem.model_validate(i) for i in row['items'])
            if sum(len(i.content) for i in items) > 2000000:
                raise ValueError('supplement text exceeds per-record budget')
            for item in items:
                if item.record_id != rid or item.evidence_id in seen:
                    raise ValueError('supplement record mismatch or duplicate evidence_id')
                if item.source_type not in _SOURCE_TYPES or set(item.metadata) - _METADATA_KEYS:
                    raise ValueError('unsupported source type or supplement metadata')
                seen.add(item.evidence_id)
            bundle._items[rid] = items
            by_id = {i.evidence_id: i for i in items}
            if not isinstance(row['images'], list) or len(row['images']) > 8:
                raise ValueError('supplement image count exceeds per-record budget')
            for image in row['images']:
                if set(image) != {'evidence_id', 'path', 'sha256', 'mime_type'}:
                    raise ValueError('image descriptor has unexpected fields')
                eid = image['evidence_id']
                item = by_id.get(eid)
                if item is None or eid in bundle._images:
                    raise ValueError('image must bind to one evidence item in its record')
                if item.metadata.get('supplemental_image_sha256') != image['sha256']:
                    raise ValueError('image and evidence hash mismatch')
                file_path = (path.parent / image['path']).resolve()
                descriptor = {**image, 'path': str(file_path), 'record_id': rid}
                raw_image = bundle._read_image_bytes(descriptor)
                with Image.open(io.BytesIO(raw_image)) as picture:
                    if (picture.format not in {'PNG','JPEG','WEBP'} or
                            getattr(picture, 'n_frames', 1) != 1 or
                            picture.width * picture.height > 25_000_000):
                        raise ValueError('unsupported or oversized supplemental image')
                    if Image.MIME[picture.format] != image['mime_type']:
                        raise ValueError('image MIME does not match decoded file')
                    picture.verify()
                with Image.open(io.BytesIO(raw_image)) as picture:
                    picture.load()
                bundle._images[eid] = descriptor
            expected_images = {i.evidence_id for i in items if 'supplemental_image_sha256' in i.metadata}
            if expected_images != {i['evidence_id'] for i in row['images']}:
                raise ValueError('image evidence lacks its original pixels')
        return bundle

    @staticmethod
    def _read_image_bytes(descriptor):
        raw = Path(descriptor['path']).read_bytes()
        if len(raw) > 10_000_000 or _sha(raw) != descriptor['sha256']:
            raise ValueError('supplemental image hash mismatch or size limit')
        return raw

    def items_for(self, record_id):
        return self._items.get(record_id, ())

    def manifest(self):
        return {'schema_version': 1, 'bundle_sha256': self.sha256,
                'information_policy': 'public_at_collection',
                'record_ids': sorted(self._items),
                'item_count': sum(map(len, self._items.values())),
                'image_count': len(self._images), 'caption_model_added': False}

    def prepare_message(self, prompt):
        """Only attach pixels for exact items visible to this role, never ID mentions."""
        try:
            payload, _ = json.JSONDecoder().raw_decode(prompt.lstrip())
        except (ValueError, TypeError):
            return prompt, ()
        if not isinstance(payload, dict):
            return prompt, ()
        view = payload.get('evidence_view', payload.get('evidence_snapshot', {}))
        if not isinstance(view, dict):
            return prompt, ()
        parts = [{'type': 'text', 'text': prompt}]
        attached, seen = [], set()
        for item in view.get('items', []):
            if not isinstance(item, dict):
                continue
            descriptor = self._images.get(item.get('evidence_id'))
            if descriptor is None or descriptor['evidence_id'] in seen:
                continue
            expected = next(i for i in self._items[descriptor['record_id']]
                            if i.evidence_id == descriptor['evidence_id'])
            if (EvidenceItem.model_validate(item) != expected or
                    view.get('record_id', expected.record_id) != expected.record_id):
                raise ValueError('visible image evidence differs from frozen bundle')
            raw = self._read_image_bytes(descriptor)
            parts.append({'type': 'text', 'text': (
                f"Original pixels for evidence_id {expected.evidence_id}. "
                "For citations to this image, use kind observation. The quote field records "
                "your visual reading, not verified verbatim text. Source identity is checked; "
                "visual interpretation and its relevance still require reasoning and review. "
                "For text evidence, continue copying exact substrings from content.")})
            parts.append({'type': 'image_url', 'image_url': {
                'url': f"data:{descriptor['mime_type']};base64,{base64.b64encode(raw).decode('ascii')}"}})
            attached.append({k: descriptor[k] for k in ('record_id','evidence_id','sha256','mime_type')})
            seen.add(descriptor['evidence_id'])
        return (parts, tuple(attached)) if attached else (prompt, ())

    def record_delivery(self, images, *, role, status):
        with self._lock:
            for rid in {i['record_id'] for i in images}:
                self._events.setdefault(rid, []).append({
                    'role': role, 'status': status,
                    'images': [dict(i) for i in images if i['record_id'] == rid],
                })

    def audit_for(self, record_id):
        with self._lock:
            events = json.loads(json.dumps(self._events.get(record_id, [])))
        return {'bundle_sha256': self.sha256,
                'supplemental_evidence_ids': [i.evidence_id for i in self.items_for(record_id)],
                'image_request_events': events,
                'delivery_note': 'response_received means the API returned; it does not prove correct image interpretation',
                'image_citation_policy': 'source_bound_visual_interpretation_not_verified_transcription'}
