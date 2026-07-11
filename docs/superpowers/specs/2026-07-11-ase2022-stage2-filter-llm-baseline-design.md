# ASE2022 Stage 1 → Stage 2 单 LLM Baseline 设计规格

## 目标

构建一个可复现的单 LLM baseline，用于预测 ASE2022 Stage 1 中的候选记录是否应保留至 Stage 2。该任务是针对特定研究的数据筛选相关性分类（study-specific fault relevance classification），而不是通用 bug 检测。

## Ground Truth

本 benchmark 使用 AutoEmpirical 当前的数据集：

- Stage 1 包含 4,184 条 ASE2022 候选记录。
- Stage 2 包含 683 条保留记录。
- 如果一条 Stage 1 记录的 `record_id` 出现在 Stage 2 中，则标记为 `accepted_fault`。
- Stage 1 中的其余记录标记为 `rejected_candidate`。

数据加载器必须验证：ASE2022 Stage 2 中的每个 `record_id` 均在 Stage 1 中恰好出现一次。如果不满足包含关系或唯一性约束，程序必须给出清晰错误并停止。

该 membership 标签近似表示原论文的人工筛选决定。由于仓库没有保留原论文自动筛选后 1,293 条 issue 的 membership，本任务不复现该中间候选集。

## 防止标签泄漏

模型只能接收以下 Stage 1 字段：

- `title`
- `body`
- `comments`
- `state`
- `created_at`

提示词及发送给模型的序列化请求不得包含下游标签或数据来源字段，包括 `symptom`、`root_cause`、`bug_type`、`component`、`sub_component`、`fix_type`、`source_file`、`source_sheet`、`original_label_json` 或任何 Stage 2 membership 信息。

为便于评估，可以在本地样本和预测结果中保存 ground-truth 标签，但不得将其放入发送给模型的消息中。

## 抽样方案

数据准备命令生成一个确定性的平衡 pilot 样本：

- 50 条 `accepted_fault` 记录。
- 50 条 `rejected_candidate` 记录。
- 使用固定随机种子，并将种子写入产物元数据。
- 排除 `title`、`body` 和 `comments` 均无有效内容的记录。

runner 还必须支持在不修改代码的情况下运行全部 4,184 条记录。报告不得将平衡 pilot 样本指标与真实类别分布下的全量指标混为一谈。

## 提示词契约

提示词应描述 ASE2022 的筛选目标，并要求模型仅在现有 issue 证据表明 JavaScript 深度学习系统中存在真实、可识别的 fault 时保留该记录。

应拒绝的情况包括：功能请求、没有系统 fault 证据的使用问题、描述不清或信息不足的 issue、无关讨论，以及命中关键词但实际上并非错误的记录。

模型必须返回严格 JSON，且只包含一个决定标签：

```json
{"decision": "accepted_fault"}
```

唯一允许的值为 `accepted_fault` 和 `rejected_candidate`。额外解释文本、格式错误的 JSON、缺失的 decision 以及未知标签均视为无效输出。

## 架构

Stage 2 filtering 使用独立、职责明确的模块和 CLI，不继续扩大现有 Stage 3 labeling 模块：

- `Benchmark/src/ase2022_stage2_filter_baseline.py`：负责数据加载、membership 标注、防泄漏提示词构建、分层抽样、输出解析、评估和运行编排。
- `Benchmark/scripts/prepare_ase2022_stage2_filter_baseline.py`：生成确定性样本和提示词产物。
- `Benchmark/scripts/run_ase2022_stage2_filter_baseline.py`：配置模型和 API transport，并执行或续跑预测。
- `tests/test_ase2022_stage2_filter_baseline.py`：验证领域逻辑。
- `tests/test_run_ase2022_stage2_filter_baseline.py`：验证 CLI 配置和 runner 集成。

为了避免重复，可以显式导入并复用现有 Stage 3 的 HTTP transport helper；但 Stage 2 filtering 必须提供独立的任务接口和实验产物。

## 实验产物

默认输出目录为：

`Benchmark/results/ase2022_stage2_filter_baseline/`

数据准备阶段生成：

- 平衡样本 CSV，包含标识字段、安全证据字段和 ground truth。
- anchored-prompts JSONL 文件。
- manifest JSON，记录样本数量、随机种子、抽样规则、源文件路径和标签语义。

每个模型运行后生成：

- prediction JSONL，包含 record ID、ground truth、原始响应、解析后的 decision、有效性、错误详情和重试元数据。
- metrics JSON，包含运行配置和评估结果。

断点续跑以 `record_id` 标识已完成记录。如果已有预测中存在重复或冲突记录，程序必须明确报错，不得静默选择其中一条。

## 评估指标

评估以 `accepted_fault` 为正类，指标包括：

- Accuracy
- Precision
- Recall
- F1
- Specificity
- Balanced accuracy
- True positives、false positives、true negatives 和 false negatives
- 有效与无效输出数量
- 无效输出率

分类指标仅在有效预测上计算。metrics 产物必须明确记录有效预测分母，防止将无效响应误解为负类预测。

## 错误处理

遇到以下情况时，程序必须给出清晰错误并停止：

- Stage 1 或 Stage 2 中 `record_id` 缺失或重复
- Stage 2 记录不存在于 Stage 1
- 出现不支持的标签
- 不存在符合条件的抽样记录
- 请求的某类样本数超过该类符合条件的记录数
- 断点续跑文件中存在重复记录
- API 配置缺失

临时模型或 transport 故障沿用现有的有限次数重试机制。重试耗尽的调用作为无效预测持久化，使实验保持可审计且能够断点续跑。

## 测试与验证

实现遵循测试驱动开发。测试必须证明：

- 能根据 membership 正确生成标签和类别数量
- 血缘关系损坏或 `record_id` 重复时会失败
- 模型提示词不包含任何禁止字段
- 50/50 分层抽样具有确定性
- 能严格解析接受和拒绝结果
- 格式错误或不支持的输出会标记为无效
- 混淆矩阵和派生指标计算正确
- 断点续跑不会产生重复模型调用
- CLI 默认值和显式覆盖参数正确

完成实现前，必须通过本任务的测试、现有 Stage 3 baseline 测试以及仓库完整测试套件。还必须执行一次本地数据准备 smoke test，并验证生成的 100 条样本中两个类别各有 50 条。

## 文档用语

仓库文档和实验报告应将本任务称为“ASE2022 Stage 2 filtering”或“针对特定研究的数据筛选相关性分类”。如果使用更简短的“bug detection”，必须同时说明其操作性定义：正类表示记录属于从原论文衍生的 Stage 2 数据集。
