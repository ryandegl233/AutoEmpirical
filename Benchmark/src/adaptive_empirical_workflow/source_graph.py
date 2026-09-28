"""Deterministic source structure and BM25 context. No model or semantic edges."""
import hashlib
import math
import re
from collections import Counter, defaultdict

VERSION = 'source-graph-v1'
URL = re.compile(r'https?://[^\s<>"\]]+')


def tokens(text):
    return re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', text.lower())


def build_source_graph(view):
    nodes, edges = {}, set()
    by_uri = defaultdict(list)
    for item in view.items:
        by_uri[item.source_uri].append(item.evidence_id)
        nodes[item.evidence_id] = dict(id=item.evidence_id, kind=item.source_type,
            evidence_id=item.evidence_id, source_uri=item.source_uri,
            author=item.metadata.get('source_author'), published_at=item.metadata.get('source_published_at'))
        # Offsets always refer to original content; never rewrite or summarize it.
        spans = list(re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', item.content))
        for span in spans:
            for start in range(span.start(), span.end(), 1000):
                end = min(start+1000, span.end())
                pid = f'{item.evidence_id}:p:{start}:{end}'
                nodes[pid] = dict(id=pid,kind='paragraph',evidence_id=item.evidence_id,
                    start=start,end=end,text=item.content[start:end])
                edges.add((item.evidence_id,pid,'contains'))
    for item in view.items:
        for url in sorted(set(m.group().rstrip('.,;)') for m in URL.finditer(item.content))):
            targets = by_uri.get(url)
            if not targets:
                uid = 'url:' + hashlib.sha256(url.encode()).hexdigest()[:20]
                nodes.setdefault(uid,dict(id=uid,kind='external_reference',source_uri=url,
                    availability='not_in_current_evidence'))
                targets = [uid]
            for target in targets:
                if target != item.evidence_id:
                    edges.add((item.evidence_id,target,'links_to'))
        reply = item.metadata.get('in_reply_to_evidence_id')
        if isinstance(reply,str) and reply in nodes:
            edges.add((item.evidence_id,reply,'reply_to'))
        # Explicit block quotes only. Ambiguous or very short matches get no edge.
        for quote in re.findall(r'(?m)^>\s?(.+)$',item.content):
            matches = [other.evidence_id for other in view.items
                       if other.evidence_id!=item.evidence_id and len(quote)>=40 and quote in other.content]
            if len(matches)==1:
                edges.add((item.evidence_id,matches[0],'quotes'))
    return dict(version=VERSION,record_id=view.record_id,
        nodes=[nodes[k] for k in sorted(nodes)],
        edges=[dict(source=s,target=t,relation=r) for s,t,r in sorted(edges)])


def source_context(view, query, *, mode='graph', top_k=6, max_nodes=24):
    if mode not in {'graph','flat'}:
        raise ValueError('source context mode must be graph or flat')
    graph = build_source_graph(view)
    nodes = {n['id']:n for n in graph['nodes']}
    docs = {k:Counter(tokens(n['text'])) for k,n in nodes.items() if n['kind']=='paragraph'}
    query_terms = set(tokens(query))
    df = Counter(t for doc in docs.values() for t in doc)
    avg = sum(sum(d.values()) for d in docs.values())/max(len(docs),1)
    scores = {}
    for key, doc in docs.items():
        length = sum(doc.values())
        scores[key] = sum(math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5)) *
            doc[t]*2.2/(doc[t]+1.2*(.25+.75*length/max(avg,1)))
            for t in query_terms if doc[t])
    seeds = sorted((k for k in docs if scores[k]>0),key=lambda k:(-scores[k],k))[:top_k]
    # Empty retrieval is NOT evidence of absence. The full original view stays in every prompt.
    selected = list(seeds)
    parents = [nodes[k]['evidence_id'] for k in seeds]
    for key in parents:
        if key not in selected: selected.append(key)
    neighbors = defaultdict(set)
    for e in graph['edges']:
        neighbors[e['source']].add(e['target']); neighbors[e['target']].add(e['source'])
    for key in list(selected):
        for neighbor in sorted(neighbors[key]):
            if neighbor not in selected and len(selected)<max_nodes:
                selected.append(neighbor)
    selected = set(selected)
    result = dict(version=VERSION,record_id=view.record_id,
        instruction='Navigation index only. Cite original evidence_id and exact source text. '
            'No edge asserts support, refutation or causation. All original evidence_view items remain available; '
            'empty retrieval does not establish absence. External references are not fetched.',
        retrieval=dict(query=query,seed_ids=seeds,bm25_k1=1.2,bm25_b=.75,
            total_nodes=len(nodes),selected_nodes=len(selected),full_original_evidence_retained=True),
        nodes=[nodes[k] for k in sorted(selected)],
        edges=[e for e in graph['edges'] if e['source'] in selected and e['target'] in selected])
    if mode=='flat':
        import json
        result['serialized_graph']=json.dumps({'nodes':result.pop('nodes'),'edges':result.pop('edges')},ensure_ascii=False)
    return result


def case_query(view):
    """Case-local deterministic seed query; never derives terms from GT or other cases."""
    primary = next((i for i in view.items if i.source_type=='record_summary'),None)
    primary = primary or next((i for i in view.items if i.source_type=='issue_body'),None)
    if primary is None: return ''
    text = primary.content
    title = re.search(r'(?m)^TITLE:\s*(.+)',text)
    return (title.group(1)+' ' if title else '')+text[:800]
