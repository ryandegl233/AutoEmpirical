# 实验二运行说明

入口为 `tools/run_experiment_two_sharded.py`。输入来自
`Benchmark/inputs/ase2022_dev48/experiment_two.json`，不再读取旧实验的
`run_manifest.json` 或 `reports/` 下的结果。配置包含来源文件、哈希、固定顺序和模型名称；
连接凭据仍放在环境变量或本地 `.env`。模型名称是配置值，不保证供应商持续提供该型号。

## 离线检查与运行

在仓库根目录执行。默认只检查配置并生成本地 dry-run 记录，不调用模型：

```powershell
python tools/run_experiment_two_sharded.py --batch-id exp2-check
```

确认模型和连接配置后，用 `--run` 启用模型调用：

```powershell
python tools/run_experiment_two_sharded.py --batch-id exp2-new --run --case-workers 2
python tools/run_experiment_two_sharded.py --batch-id exp2-new --run --resume
python tools/run_experiment_two_sharded.py --batch-id exp2-new --score-only
```

自定义输入使用 `--input-config <path>`。路径相对于该配置文件所在目录，所有输入及图片
都须位于该目录内，并有匹配的哈希。使用新的配置版本和 batch ID；不要覆盖冻结输入。
全量为四组各 48 例。`--cases` 可选择探索子集，不能混入全量结果。

`--case-workers` 控制并发案例数，默认 1；每个案例内部仍可并发请求。
同一案例轮换组别顺序，不同案例使用独立子进程和输出目录。
并发数、代码、输入和图片绑定到本地冻结协议；变更后拒绝续跑。
已完成案例包括无效输出都会保留，不自动反复重跑直到有效。

## 四组设置

| 组别 | 内容 |
| --- | --- |
| E00 | rules-v4 对照 |
| E10 | E00 加前置证据核对、解释修正及裁决解释检查 |
| E01 | E00 加确定性来源图、BM25 和相邻来源检索 |
| E11 | E10 加来源图 |

规则不包含案例答案，图不生成因果关系。原文和图片通过相同输入通道提供；
四组使用同一角色预算，实际调用量可因重试、返工或提前结束而不同。
`--graph-serialization flat` 只改变同一检索结果的表示形式，需独立 batch。

48 例是已检查的开发数据；另外两例保留作方法示例。评分端读取原 50 条 gold，
仅评分固定 48 例，gold 路径不传给模型 worker。无效输出保留在分母内。
联合/分项准确率、配对变化和区间由本地评分脚本计算；单批案例重采样区间
不代表多次模型运行的波动。不同 provider、续跑方式和策略版本分别报告。

## 本地输出

结果保存在 `Benchmark/runs/experiment_two/<batch-id>/`，离线检查使用 `-check` 后缀。
包括冻结协议、分案例预测、请求追踪、日志及评分表。全部被 Git 忽略，
不随仓库或 release 附件上传。`tools/watch_experiment_two.py` 可查看本地进度。

UTF-8 和 relay 入口保留在 `tools/`，各自记录传输与续跑策略。
请求中的模型名称不能证明网关实际调用了同一个上游模型。

## A/B 方法实验

A/B 使用相同 48 例但不同协议，不能与 E00–E11 混为一组：

```powershell
./Benchmark/scripts/ase2022_dev48/run_four_arms.ps1 -DryRun -Full -BatchId ab-check
./Benchmark/scripts/ase2022_dev48/run_four_arms.ps1 -Run -Full -BatchId ab-new -Model <model-id>
```

默认离线；`-Run` 显式启用模型调用。结果位于 `Benchmark/runs/ase2022_dev48/`。
方法示例重建程序为 `Benchmark/scripts/ase2022_dev48/build_examples.py`；
指定新的本地 `--output-dir`，不会覆盖现有冻结版本。
