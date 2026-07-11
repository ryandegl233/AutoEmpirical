# ASE2022 Stage 2 Filtering 单 LLM Baseline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 构建一个以 Stage 2 `record_id` membership 为 ground truth、无下游标签泄漏、支持固定 50/50 pilot 样本和全量运行的 ASE2022 Stage 1 → Stage 2 单 LLM baseline。

**架构：** 新建独立领域模块，负责加载、标注、抽样、提示词、解析、指标和运行编排；两个轻量 CLI 分别负责准备实验输入和调用模型。复用现有 Stage 3 runner 中已验证的 HTTP transport helper，但不复用其 Stage 3 标签逻辑。

**技术栈：** Python 3、pandas、pytest、JSON/JSONL、现有 OpenAI-compatible HTTP transport。

## 全局约束

- 正类固定为 `accepted_fault`，负类固定为 `rejected_candidate`。
- 模型输入仅允许 `title`、`body`、`comments`、`state`、`created_at`。
- 默认 pilot 样本为固定种子的 50 个正类和 50 个负类。
- 分类指标只在有效预测上计算，并单独报告 invalid-output rate。
- 不修改或删除用户当前工作区中的既有实验结果和未提交变更。

---

### Task 1：数据加载、membership 标签与确定性抽样

**文件：**
- 新建：`Benchmark/src/ase2022_stage2_filter_baseline.py`
- 新建：`tests/test_ase2022_stage2_filter_baseline.py`

**接口：**
- 生成 `load_ase2022_stage2_filter_examples(stage1_path, stage2_path) -> list[dict[str, str]]`
- 生成 `select_balanced_sample(examples, per_class=50, seed=20260711) -> list[dict[str, str]]`
- 每条 example 包含标识、安全文本字段和 `decision`，不包含下游标签。

- [ ] **Step 1：编写失败测试**

测试使用临时 CSV 构造三个 Stage 1 候选和一个 Stage 2 正类，断言 membership 标签正确、返回字段严格受限；另写重复 ID、Stage 2 不属于 Stage 1、固定抽样和样本不足测试。

- [ ] **Step 2：运行测试并确认按预期失败**

运行：`python -m pytest tests/test_ase2022_stage2_filter_baseline.py -q`

预期：因模块或函数尚不存在而失败。

- [ ] **Step 3：实现最小数据逻辑**

实现常量：

```python
ACCEPTED_FAULT = "accepted_fault"
REJECTED_CANDIDATE = "rejected_candidate"
SAFE_TEXT_FIELDS = ("title", "body", "comments", "state", "created_at")
```

加载两个 CSV、限定 ASE2022 paper ID、校验 `record_id` 非空且唯一、校验 Stage 2 是 Stage 1 子集，仅复制安全字段并由 membership 生成 `decision`。抽样函数按 decision 分组，使用两个从同一 seed 派生的本地随机数生成器抽取并最终按 `record_id` 排序。

- [ ] **Step 4：运行测试并确认通过**

运行：`python -m pytest tests/test_ase2022_stage2_filter_baseline.py -q`

预期：Task 1 测试全部通过。

### Task 2：提示词、严格解析和二分类指标

**文件：**
- 修改：`Benchmark/src/ase2022_stage2_filter_baseline.py`
- 修改：`tests/test_ase2022_stage2_filter_baseline.py`

**接口：**
- 生成 `build_system_prompt() -> str`
- 生成 `build_user_prompt(example) -> str`
- 生成 `parse_filter_response(raw_text) -> dict[str, object]`
- 生成 `evaluate_filter_predictions(examples, predictions) -> dict[str, int | float]`

- [ ] **Step 1：编写提示词和解析失败测试**

断言 prompt 包含筛选语义和安全 evidence，但不包含 ground truth、下游字段名或其值；断言两个合法 JSON 被接受，markdown fence、额外 key、未知标签、额外文本和 malformed JSON 均无效。

- [ ] **Step 2：运行测试并确认失败原因是接口缺失**

运行：`python -m pytest tests/test_ase2022_stage2_filter_baseline.py -q`

- [ ] **Step 3：实现提示词与严格解析**

system prompt 明确纳入/排除标准并要求唯一 JSON key `decision`。解析器先执行 `json.loads(raw_text.strip())`，再验证结果是 key 集合恰为 `{"decision"}` 的对象且标签属于允许集合；返回 `decision`、`invalid` 和 `error`。

- [ ] **Step 4：编写指标失败测试**

使用 TP、FP、TN、FN 各一条和一条 invalid 预测，断言 confusion matrix、accuracy、precision、recall、F1、specificity、balanced accuracy、valid denominator 和 invalid rate 的精确值。

- [ ] **Step 5：实现指标并运行测试**

分类指标分母为 valid predictions；所有除零情况返回 `0.0`。invalid rate 分母为全部待评估记录。

运行：`python -m pytest tests/test_ase2022_stage2_filter_baseline.py -q`

预期：Task 1–2 测试全部通过。

### Task 3：实验产物准备脚本

**文件：**
- 修改：`Benchmark/src/ase2022_stage2_filter_baseline.py`
- 新建：`Benchmark/scripts/prepare_ase2022_stage2_filter_baseline.py`
- 修改：`tests/test_ase2022_stage2_filter_baseline.py`

**接口：**
- 生成 `write_sample_csv(examples, output_path) -> Path`
- 生成 `write_prompts_jsonl(examples, output_path) -> Path`
- 生成 `write_manifest(all_examples, selected_examples, output_path, seed, per_class, source_paths) -> Path`

- [ ] **Step 1：编写产物失败测试**

断言 CSV 为 100 条且类别平衡；prompts JSONL 中发送给模型的 `system_prompt` 与 `user_prompt` 不含 decision；manifest 精确记录全量类别数、样本类别数、seed、源文件和标签语义。

- [ ] **Step 2：运行测试确认失败**

运行：`python -m pytest tests/test_ase2022_stage2_filter_baseline.py -q`

- [ ] **Step 3：实现 writer 和 CLI**

CLI 参数包括 `--stage1-path`、`--stage2-path`、`--output-dir`、`--per-class`、`--seed` 和 `--full`。默认生成 50/50 pilot；`--full` 使用全部 examples，但仍在 manifest 标明真实类别分布。

- [ ] **Step 4：运行 focused tests 和真实数据 smoke test**

运行：

```powershell
python -m pytest tests/test_ase2022_stage2_filter_baseline.py -q
python Benchmark/scripts/prepare_ase2022_stage2_filter_baseline.py
```

预期：输出 100 条，`accepted_fault=50`，`rejected_candidate=50`。

### Task 4：模型 runner、重试、断点续跑与 CLI

**文件：**
- 修改：`Benchmark/src/ase2022_stage2_filter_baseline.py`
- 新建：`Benchmark/scripts/run_ase2022_stage2_filter_baseline.py`
- 新建：`tests/test_run_ase2022_stage2_filter_baseline.py`

**接口：**
- 生成 `run_filter_prompts(..., model_caller=None) -> dict[str, object]`
- runner CLI 与 Stage 3 runner 一致地支持 `--models`、`--limit`、`--sleep-seconds`、`--max-retries`、`--retry-delay-seconds`、`--no-resume`、`--base-url`、`--responses-path` 和 `--http-client`。

- [ ] **Step 1：编写 runner 失败测试**

通过注入的 deterministic fake caller 测试合法结果、异常耗尽后 invalid 持久化、已有结果 resume、重复 resume record 报错、metrics 文件字段和 `--limit` 行为。

- [ ] **Step 2：运行测试确认失败**

运行：`python -m pytest tests/test_run_ase2022_stage2_filter_baseline.py -q`

- [ ] **Step 3：实现最小 runner**

读取 prompts JSONL，以 `record_id` 为 resume key；调用注入 caller 或复用 Stage 3 `call_model_with_retries`；逐条 append JSONL；最后按 prompts 顺序解析已有与新增预测并生成 metrics JSON。发现重复 `record_id` 时抛出 `ValueError`。

- [ ] **Step 4：实现 CLI 配置测试和脚本**

复用现有 `.env` 加载与 `resolve_run_config`，但使用 Stage 2 filtering 的默认 prompts、输出目录和文件名。

- [ ] **Step 5：运行 runner 和回归测试**

运行：

```powershell
python -m pytest tests/test_run_ase2022_stage2_filter_baseline.py -q
python -m pytest tests/test_ase2022_llm_baseline_runner.py tests/test_run_ase2022_llm_baseline.py -q
```

预期：新旧 baseline 测试均通过。

### Task 5：文档与完整验证

**文件：**
- 修改：`Benchmark/README.md`
- 修改：`Benchmark/scripts/README.md`
- 修改：`Benchmark/src/README.md`
- 验证：`Benchmark/results/ase2022_stage2_filter_baseline/`

- [ ] **Step 1：更新使用文档**

记录任务定义、membership ground truth、数据泄漏限制、pilot/full 命令、输出文件、指标分母和“非通用 bug detection”说明。

- [ ] **Step 2：检查生成产物**

用脚本验证 sample 和 prompts 均为 100 条、record ID 唯一、类别各 50 条、prompt 不含 decision 和禁止字段。

- [ ] **Step 3：运行完整测试套件**

运行：`python -m pytest -q`

预期：0 failures。

- [ ] **Step 4：检查工作区差异**

运行：`git diff --check` 和 `git status --short`，确认没有覆盖既有未提交结果或无关文件。
