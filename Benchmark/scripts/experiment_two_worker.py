"""Capture experiment-two requests/responses, then run the existing workflow CLI."""
import hashlib
import json
import os
import runpy
import sys
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from Benchmark.src.ase2022_llm_baseline import PooledChatCompletionClient
    directory=Path(os.environ['AE_EXPERIMENT_TWO_TRACE_DIR'])
    directory.mkdir(parents=True,exist_ok=True)
    original=PooledChatCompletionClient.call_model
    def traced(self,**kwargs):
        call_id=uuid.uuid4().hex
        user=kwargs['user_prompt']
        text=user if isinstance(user,str) else '\n'.join(p['text'] for p in user if p.get('type')=='text')
        try: payload=json.JSONDecoder().raw_decode(text.lstrip())[0]
        except ValueError: payload={}
        view=payload.get('evidence_view',payload.get('evidence_snapshot',{}))
        items=view.get('items',[])
        rid=view.get('record_id') or (items[0].get('record_id') if items else None)
        # Visible text plus image hashes; never credentials, request headers or provider internals.
        images=[] if isinstance(user,str) else [hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
            for p in user if p.get('type')=='image_url']
        request=dict(call_id=call_id,record_id=rid,team_id=payload.get('team_id'),
            created_at=datetime.now(timezone.utc).isoformat(),
            **{k:kwargs.get(k) for k in ['model','temperature','max_tokens','thinking_enabled']},
            messages=[{'role':'system','content':kwargs['system_prompt']},{'role':'user','content':text}],
            image_part_hashes=images,evidence=[{'id':i['evidence_id'],'content_sha256':hashlib.sha256(i['content'].encode()).hexdigest()} for i in items])
        (directory/f'{call_id}.request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
        event=dict(call_id=call_id,record_id=rid)
        started=time.perf_counter()
        try:
            response=original(self,**kwargs)
        except BaseException as error:
            event.update(status='transport_failed',exception_type=type(error).__name__)
            raise
        else:
            event.update(status='response_received',response_text=str(response),
                **{k:getattr(response,k,None) for k in ['prompt_tokens','completion_tokens','finish_reason','response_model']})
            return response
        finally:
            event['elapsed_seconds']=time.perf_counter()-started
            (directory/f'{call_id}.response.json').write_text(json.dumps(event,ensure_ascii=False),encoding='utf-8')
    PooledChatCompletionClient.call_model=traced
    target=Path(__file__).with_name('run_adaptive_empirical_workflow.py')
    sys.argv[0]=str(target)
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__': main()
