# 01 作业报告

## 推导检查

- `Y=XW^T+b` 的 shape 推导：
- `MLP(4,[16,16],2)` 的 shape trace：
- autograd 链式法则例子：
- dynamic batch 的边界：

## 实验记录

| 项目 | 观察值 | 解释 |
| --- | --- | --- |
| exported | | |
| max_abs_diff | | |
| mean_abs_diff | | |
| passed | | |

## 证据

- demo 代码定位：
- MLP shape 检查定位：
- export 入口定位：
- ONNX check 定位：

## 失败分析

- 我做的 ablation：
- 报错或输出：
- 对应的 tensor contract 违反：
- 恢复代码证据：

## 卡住点

- 我卡在：
- 是 shape、dtype、device 还是 autograd：
- 我用哪段代码验证：

## 评分要点自查

- 是否能逐层推 shape：
- 是否区分 eval 和 inference_mode：
- 是否解释 passed=True：
- 是否有代码定位证据：

## 下一步

- 进入 02 前我如何表示 state/action/reward：
- 我会如何检查 RL batch 的 tensor contract：
