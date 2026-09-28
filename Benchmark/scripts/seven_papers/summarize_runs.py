"""Summarize actual saved predictions, including validation failures."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Benchmark.src.paper_benchmark import read_stage, model_evidence_fields
from Benchmark.src.annotation_contracts import annotation_metrics


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def summarize(base,prepared):
    results=[]
    for folder in sorted(p for p in base.iterdir() if p.is_dir() and (prepared/p.name/'cohort.csv').exists()):
        cohort=read_stage(prepared/folder.name/'cohort.csv')
        taxonomy=json.loads((prepared/folder.name/'taxonomy.json').read_text(encoding='utf-8'))
        for engine_path in sorted(p for p in folder.iterdir() if p.is_dir()):
            engine=engine_path.name.split('_',1)[0]
            for path in sorted(engine_path.rglob('*predictions*.jsonl')):
                rows=read_jsonl(path)
                stage='stage2' if 'stage2' in str(path.relative_to(engine_path.parent)) else 'stage3'
                selected=[r for r in cohort.values() if stage=='stage2' or r['decision']=='accepted_fault']
                actual={r['record_id']:r for r in rows}
                if len(actual)!=len(rows) or not set(actual)<=set(cohort):
                    raise ValueError(f'prediction identities invalid: {path}')
                valid=[];normalized=[];errors=[];request_count=0;citations=0
                for row in rows:
                    if engine=='self':
                        okay=bool(row.get(stage+'_valid'))
                        pred={'decision':row.get('stage2_prediction'),'symptom':row.get('symptom_prediction'),
                              'root_cause':row.get('root_cause_prediction')}
                    elif engine=='mas':
                        okay=not row.get('invalid',True)
                        pred=row.get('final_prediction',{})
                        audit=row.get('explanation_audit',{})
                        if audit.get('evidence_fields') != model_evidence_fields(cohort[row['record_id']]):
                            raise ValueError(f'MAS input binding mismatch: {row["record_id"]}')
                        citations+=len(audit.get('citations',[]))
                        request_count+=len(audit.get('request_trace',[]))
                    else:
                        okay=not row.get('invalid',True)
                        pred=row
                        request_count+=row.get('attempts',0)
                    valid.append(okay)
                    normalized.append({'record_id':row['record_id'],'invalid':not okay,
                                       **{d:pred.get(d) for d in ('symptom','root_cause')}})
                    if not okay:
                        verification=(row.get('audit') or {}).get(stage) or {}
                        errors.append({'record_id':row['record_id'],'error':row.get('error'),
                            'verification':verification.get('verification'),
                            'unresolved':verification.get('unresolved')})
                result={'domain':folder.name,'engine':engine,'stage':stage,'expected_records':len(selected),
                    'saved_records':len(rows),'valid_records':sum(valid),'invalid_or_missing':len(selected)-sum(valid),
                    'predictions_path':str(path.resolve()),'observed_request_records':request_count,
                    'citations':citations,'invalid_details':errors}
                if stage=='stage3':
                    result['annotation_metrics']=annotation_metrics(selected,normalized,taxonomy)
                else:
                    correct=0
                    for row,okay in zip(rows,valid):
                        prediction=row.get('stage2_prediction') if engine=='self' else row.get('final_prediction',{}).get('decision') if engine=='mas' else row.get('decision')
                        correct+=bool(okay and prediction==cohort[row['record_id']]['decision'])
                    result['cohort_membership_accuracy']=correct/len(selected) if selected else None
                    result['metric_target']='agreement_with_paper_selected_cohort'
                results.append(result)
    return {'scope':'actual saved outputs; validation failure is retained, not called successful classification',
            'rows':results,'saved_predictions':sum(r['saved_records'] for r in results),
            'valid_predictions':sum(r['valid_records'] for r in results),
            'invalid_or_missing':sum(r['invalid_or_missing'] for r in results),
            'mas_citations':sum(r['citations'] for r in results),
            'observed_single_and_mas_request_records':sum(r['observed_request_records'] for r in results)}


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    base=Path(sys.argv[1]);prepared=Path(sys.argv[2]) if len(sys.argv)>2 else base/'inputs'
    report=summarize(base,prepared)
    (base/'prediction_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
