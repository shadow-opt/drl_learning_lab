# 07 Experiment Engineering

## 学习目标

本章目标是让每次训练都可复现、可检查、可比较、可失败复盘。学完后你应该知道一次 DRL 实验至少要保存哪些 artifact，如何区分 train 和 eval，如何记录 seed、config、environment、metrics、checkpoint、export，以及如何写失败报告。

这章不是算法附属品。DRL 训练波动大、调参敏感、环境随机性强，没有实验工程，你很难判断一次提升是算法有效、seed 偶然，还是 bug 造成的假象。

## 为什么重要

监督学习里，数据集固定，指标相对稳定；DRL 里，数据分布由当前 policy 产生，训练过程本身会改变未来样本。一次训练失败可能来自 reward scale、探索、done 处理、target network、seed、环境版本、导出路径等任意环节。

实验工程的价值是把“我感觉这次训练好了”变成可检查证据：配置是什么、环境是什么、指标怎么变、模型文件在哪里、是否能 eval、是否能导出、失败时定位过什么。

## 核心直觉

一个 run 应该像一个可复盘的实验包，而不是终端里消失的一串输出。最小 run 产物应包括：

```text
config.json
environment.json
metrics.csv
checkpoint
eval result
export artifact
report
```

其中 config 记录“我想怎么跑”，environment 记录“实际在哪跑”，metrics 记录“过程中发生了什么”，checkpoint 记录“模型状态”，report 记录“我理解了什么”。

最小 artifact 判定表：

```text
config.json       能否知道超参数和 seed
environment.json  能否知道依赖版本和运行环境
metrics.csv       能否画出训练过程
checkpoint        能否恢复模型状态
*.onnx            能否离开训练循环推理
report.md         能否复盘目标、失败和结论
```

## 数学与公式

实验比较至少要看多个 episode 的平均 return：

$$
\bar{R}=\frac{1}{N}\sum_{i=1}^{N}R_i
$$

还要注意方差：

$$
\sigma_R^2=\frac{1}{N}\sum_{i=1}^{N}(R_i-\bar{R})^2
$$

单个 episode 高分不能证明算法有效。对自学仓库来说，不要求严格论文级统计，但至少要避免用一次偶然成功判断实现正确。

## 算法流程

一次训练实验的工程流程：

1. 固定 seed。
2. 构造 config，并保存 `config.json`。
3. 保存 Python、PyTorch、ONNXRuntime 等环境信息。
4. 创建 run directory。
5. 训练时持续写 metrics。
6. 定期保存 checkpoint。
7. 使用独立 eval 逻辑评估。
8. 导出 inference 模型。
9. 写 report，总结结果和失败。
10. 跑 tests/quality gates 后提交。

失败实验也要写 report。因为失败记录能帮助你建立 debug 索引，避免下次重复踩坑。

自学时建议给每次 run 写一个很短的判断：这次是“实现验证”、“超参数观察”还是“失败定位”。实现验证只需要短步数和 smoke test；超参数观察需要固定其他变量；失败定位需要一次只改一个假设。不要把这些目标混在同一次 run 里，否则 report 很难得出结论。

## 代码双索引

- Demo：`labs/07_experiment_engineering/code/snapshot_demo.py`
- Experiment helpers：`src/drl_lab/common/experiment.py`
- Checkpoint helpers：`src/drl_lab/common/checkpoint.py`
- Logging helpers：`src/drl_lab/common/logging.py`
- Snapshot test：`tests/test_experiment_snapshots.py`
- 通用模板：`docs/experiment_report_template.md`
- Debug 清单：`docs/debug_checklist.md`

训练入口示例：`src/drl_lab/algorithms/dqn/train.py`、`src/drl_lab/algorithms/sac/train.py`。

## 实验任务

```bash
conda run -n drl-lab python labs/07_experiment_engineering/code/snapshot_demo.py
```

然后跑一个短训练：

```bash
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
```

检查 run 目录中是否存在 config、environment、metrics、checkpoint、ONNX。把发现写入 `labs/07_experiment_engineering/report.md`。

## Debug Checklist

- seed 是否写入 config。
- run directory 是否唯一且可追踪。
- metrics 是否包含 step 和关键指标。
- checkpoint 是否能 reload。
- eval 是否关闭探索噪声。
- export artifact 是否对应当前 checkpoint。
- 失败时是否记录现象、假设、定位和修复。
- `experiments/runs/` 是否没有被误提交。
- 是否能从 report 反推出当时运行的命令。
- 是否记录了“没有改善”的实验，而不只记录成功实验。

## Spinning Up 对照

Spinning Up 提供 logger 和实验组织思路，但本仓库要求更完整的工程闭环：config/environment snapshot、checkpoint、eval、ONNX export、ONNXRuntime consistency 和中文 report。也就是说，本仓库不只问“算法能否训练”，还问“这次训练能否复现和交付”。

## 学完标准

- 能列出一次 run 必须保存的 artifact。
- 能解释 config、metrics、checkpoint、report 的区别。
- 能跑 snapshot demo 和一次短训练。
- 能打开 run 目录判断一次实验是否可复盘。
- 能用 `docs/debug_checklist.md` 记录一次失败定位。
- 能根据 report 复现同一次命令和关键配置。
- 能区分实现验证、超参数观察和失败定位三类 run。
