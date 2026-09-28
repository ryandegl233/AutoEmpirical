import hashlib
import json

import pytest

from Benchmark.src import ase2022_llm_baseline as baseline
from tests.test_supplemental_evidence import fixture_bundle, bundle_class


def test_pooled_response_retains_finish_reason_and_returned_model(monkeypatch):
    class Response:
        status = 200
        def read(self):
            return json.dumps({'model': 'returned-version', 'choices': [
                {'message': {'content': '{"ok":true}'}, 'finish_reason': 'length'}
            ], 'usage': {'prompt_tokens': 12, 'completion_tokens': 20}}).encode()
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args, **kwargs): pass
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(baseline.http.client, 'HTTPSConnection', Connection)
    client = baseline.PooledChatCompletionClient(base_url='https://example.org/v1', max_connections=1)
    try:
        response = client.call_model(system_prompt='s', user_prompt='u', model='requested',
                                     api_key='test', wire_api='chat_completions')
        assert response == '{"ok":true}'
        assert response.finish_reason == 'length'
        assert response.response_model == 'returned-version'
        assert response.prompt_tokens == 12
    finally:
        client.close()


@pytest.mark.parametrize('length,accepted', [(1500000, True), (2000001, False)])
def test_full_model_text_is_preserved_or_rejected_never_sliced(tmp_path, length, accepted):
    path, _, _, payload = fixture_bundle(tmp_path)
    item = payload['records'][0]['items'][0]
    text = 'model-node;' * (length // 11) + 'x' * (length % 11)
    assert len(text) == length
    item.update(content=text, content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                source_type='source_code', metadata={})
    payload['records'][0]['images'] = []
    path.write_text(json.dumps(payload), encoding='utf-8')
    if not accepted:
        with pytest.raises(ValueError, match='per-record budget'):
            bundle_class().load(path, allowed_record_ids={'case-one'})
        return
    bundle = bundle_class().load(path, allowed_record_ids={'case-one'})
    assert bundle.items_for('case-one')[0].content == text
