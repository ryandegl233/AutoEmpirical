# ASE2022 单 LLM Baseline 实验结果汇总

_汇总 Stage 1 → Stage 2 filtering 与 Stage 2 → Stage 3 labeling 的固定样本实验；生成日期：2026-07-13。_

---

## 📊 Stage 1 → Stage 2 filtering

实验使用100条平衡样本，其中50条为 `accepted_fault`，50条为 `rejected_candidate`。表中结果使用支持 Markdown fenced JSON 的容错解析器重新计算；Claude 有1条、Gemini有3条输出经过格式归一化，归一化后所有模型均有100条有效预测。

| 模型 | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 | Specificity | Balanced accuracy | 格式归一化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DeepSeek Chat** | 45 | 26 | 24 | 5 | **69.0%** | 63.4% | **90.0%** | **74.4%** | 48.0% | **69.0%** | 0 |
| **GPT-4o-mini** | 37 | 23 | 27 | 13 | 64.0% | 61.7% | 74.0% | 67.3% | 54.0% | 64.0% | 0 |
| **Claude 3.5 Sonnet** | 29 | 19 | **31** | 21 | 60.0% | 60.4% | 58.0% | 59.2% | **62.0%** | 60.0% | 1 |
| **Gemini 2.5 Flash** | 34 | 25 | 25 | 16 | 59.0% | 57.6% | 68.0% | 62.4% | 50.0% | 59.0% | 3 |
| **o3** | 29 | 26 | 24 | 21 | 53.0% | 52.7% | 58.0% | 55.2% | 48.0% | 53.0% | 0 |

### 汇报要点

- DeepSeek Chat 获得最高 Recall、F1、Accuracy 和 Balanced accuracy，但将71条记录预测为正类，误报相对较多
- GPT-4o-mini 的 Recall、Specificity 和 F1 相对均衡
- Claude 3.5 Sonnet 的 Specificity 最高，但漏掉21条真实 fault
- o3 在当前提示词和样本上的 Accuracy 为53%，仅略高于平衡样本的50%基准

## 📊 Stage 2 → Stage 3 labeling

实验使用固定50条样本，同时预测 `symptom` 与 `root_cause`。分类准确率以有效输出为分母；Claude 的7条 invalid 不进入其分类准确率分母。

| 模型 | 总数 | 有效 | Invalid | Invalid rate | Symptom accuracy | Root-cause accuracy | Joint exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Claude 3.5 Sonnet** | 50 | 43 | 7 | 14.0% | **76.7%** | 44.2% | 34.9% |
| **Gemini 2.5 Flash** | 50 | 50 | 0 | 0.0% | 76.0% | 46.0% | 32.0% |
| **o3** | 50 | 50 | 0 | 0.0% | 72.0% | **50.0%** | **38.0%** |
| **DeepSeek Chat** | 50 | 50 | 0 | 0.0% | 72.0% | 40.0% | 32.0% |
| **GPT-4o-mini** | 50 | 50 | 0 | 0.0% | 66.0% | 26.0% | 16.0% |

### 汇报要点

- Claude 的 valid-only Symptom accuracy 最高，但其14% invalid rate 会使该结果偏乐观；按全部50条计算的端到端 Symptom accuracy 为66.0%
- o3 获得最高 Root-cause accuracy 与 Joint exact match，且没有 invalid 输出
- Gemini 的 Symptom accuracy 接近 Claude，同时保持100%有效输出
- GPT-4o-mini 在 Stage 3 的 Root-cause accuracy 和 Joint exact match 最低

## 🔎 指标口径

| 指标 | 含义 |
| --- | --- |
| **Accuracy** | 有效预测中，正类与负类整体预测正确的比例 |
| **Precision** | 模型预测为 `accepted_fault` 的记录中，真实属于 Stage 2 的比例 |
| **Recall** | 真实属于 Stage 2 的记录中，被模型成功保留的比例 |
| **F1** | Precision 与 Recall 的调和平均 |
| **Specificity** | 真实负样本中，被模型正确拒绝的比例 |
| **Balanced accuracy** | Recall 与 Specificity 的平均值 |
| **Symptom accuracy** | 有效 Stage 3 输出中，`symptom` 标签完全正确的比例 |
| **Root-cause accuracy** | 有效 Stage 3 输出中，`root_cause` 标签完全正确的比例 |
| **Joint exact match** | `symptom` 与 `root_cause` 同时完全正确的比例 |
| **Invalid rate** | 不符合输出契约、无法进入分类指标计算的响应比例 |

## ⚠️ 使用限制

- Stage 2 表基于50/50平衡样本，不能直接代表4,184条全量数据上的真实 Precision 或 Accuracy
- Stage 3 表仅包含50条样本；一个样本对应约2个百分点，不应将很小的模型差距解释为显著差异
- Claude 的 Stage 3 分类准确率使用43条有效输出作为分母；汇报时必须同时展示14% invalid rate
- Stage 2 的 Markdown fence 归一化只移除单个标准代码块，不从任意解释文本中猜测或抽取答案
