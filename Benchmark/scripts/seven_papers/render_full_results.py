"""Render a fixed-cohort readout only after completion and independent audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def percentage(value):
    return '不评分' if value is None else f'{100*value:.1f}%'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root
    completion=read(root/'completion.json')
    summary=read(root/'prediction_verification.json')
    audit=read(root/'independent_evidence_audit.json')
    assert completion['status']=='PASS' and not completion['source_drift']
    assert summary['saved_predictions']==1500 and len(summary['rows'])==20
    assert audit['status']=='PASS' and audit['total_saved']==1500 and audit['total_missing']==0
    rows=[]
    for domain in ('fse2021','icse2021','icse2022','icse2023','icse2024'):
        for engine in ('mas','self'):
            stage2=next(r for r in summary['rows'] if (r['domain'],r['engine'],r['stage'])==(domain,engine,'stage2'))
            stage3=next(r for r in summary['rows'] if (r['domain'],r['engine'],r['stage'])==(domain,engine,'stage3'))
            assert stage2['saved_records']==100 and stage3['saved_records']==50
            metrics=stage3['annotation_metrics']
            rows.append({'domain':domain,'engine':engine,'filter_valid':stage2['valid_records'],
                'filter_n':100,'filter_selected_cohort_agreement':stage2['cohort_membership_accuracy'],
                'annotation_valid':stage3['valid_records'],'annotation_n':50,
                'symptom_accuracy':metrics['symptom_accuracy'],'root_cause_accuracy':metrics['root_cause_accuracy'],
                'joint_accuracy':metrics['joint_accuracy'],'symptom_micro_f1':metrics.get('symptom_micro_f1'),
                'filter_predictions_path':stage2['predictions_path'],'annotation_predictions_path':stage3['predictions_path']})
    with (root/'result_table.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    lines=['# 新五篇完整实验结果（v2）','',
        f"已保存 1,500/1,500 条预测，其中 {summary['valid_predictions']} 条通过输出验证；"
        f"{summary['invalid_or_missing']} 条失败或未解决。独立证据审计 {audit['check_count']:,} 项通过。",'',
        '每篇每引擎固定 100 条 filter、50 条 annotation；annotation 样本独立于 filter 预测。'
        '无效输出保留在分母中，不用重跑结果替换。','',
        '| 论文 | 引擎 | filter 有效 | 入选集一致性 | annotation 有效 | 症状 | 根因 | 联合 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['domain']} | {r['engine']} | {r['filter_valid']}/100 | "
            f"{percentage(r['filter_selected_cohort_agreement'])} | {r['annotation_valid']}/50 | "
            f"{percentage(r['symptom_accuracy'])} | {percentage(r['root_cause_accuracy'])} | {percentage(r['joint_accuracy'])} |")
    lines+=['',
        '**解释边界：** PyTorch 的 filter 单列作者入选集一致性，未入选不等于不是 bug。'
        'IoT 症状按标签集合精确匹配；UAV 自由文本和 DL 恒定症状不计算症状/联合分类准确率。'
        '不同论文的任务和类别不同，不合并成一个跨论文准确率。','',
        '**IoT 多标签补充：** '+ '；'.join(f"{r['engine']} micro-F1={percentage(r['symptom_micro_f1'])}" for r in rows if r['domain']=='icse2021')+'。','',
        '来源核查证明记录和引用可追溯，不等同于证明每条引用在语义上支持模型结论。'
        'MAS 保存明确理由、精确引文和请求/响应；Self 保存角色报告与证据账本，预测文件没有原始 HTTP 追踪。','',
        '逐样本失败和有效性见 `prediction_verification.json`；来源、引文、记录绑定见 '
        '`independent_evidence_audit.json`；冻结命令与源代码见 `run_plan.json`、`source_snapshot.zip`。'
        '本次不混入已中止的 v1、预跑或历史 T1/T2 结果。','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    inputs=('completion.json','prediction_verification.json','independent_evidence_audit.json','result_table.csv','RESULTS.md')
    (root/'readout_sha256.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in inputs},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'saved':1500,'valid':summary['valid_predictions'],'report':str((root/'RESULTS.md').resolve())}))


if __name__=='__main__':main()
