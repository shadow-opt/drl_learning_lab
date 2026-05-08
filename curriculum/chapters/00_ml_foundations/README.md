# 00 ML Foundations

Status: `beginner-ready`

本章是 DRL 课程的监督学习和优化预备课。它不是压缩版机器学习综述，而是一组可验证的公开课式 lecture：从数据、模型、loss、gradient、optimizer 到泛化与实验记录，建立后续 DQN、policy gradient、actor-critic 都会复用的训练语言。

## 先修数学

- Python：函数、类、列表、循环、简单文件路径。
- 线性代数：向量、矩阵乘法、转置、矩阵 shape。
- 微积分：一元导数、链式法则、梯度是多变量导数的向量。
- 概率：均值、方差、训练集/验证集抽样直觉。

不熟也可以学：本章会在 `lesson.md` 内补齐必要定义和手算例题。

## 预计用时

- Lecture notes：2.5-3.5 小时。
- Code walkthrough：1.5-2 小时。
- Lab：1 小时。
- Problem Set：3-5 小时。
- Report：30-45 分钟。

## Lecture Map

1. 监督学习形式化：dataset、model、parameter、prediction、target。
2. 经验风险：MSE、BCE、CrossEntropy 为什么都是 loss。
3. 梯度下降：从一元二次函数到矩阵线性回归。
4. Batch SGD/Adam：为什么训练循环必须按固定顺序写。
5. 泛化：overfit、validation、bias-variance、指标解释。
6. 工程闭环：metrics、checkpoint、ONNX consistency。

## Problem Set

完成 `exercises.md` 中至少：

- 全部概念题。
- 全部手算题。
- 至少 2 道推导题。
- 全部代码题。
- 至少 2 道实验诊断题。

卡住先看 `hints.md`，最后用 `solutions.md` 对照评分要点。

## Demo

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/binary_classifier.py
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/image_classifier.py
```

## 完成标准

1. 能用公式和白话同时解释 `forward -> loss -> zero_grad -> backward -> step`。
2. 能手算简单 MSE 和一元梯度下降更新。
3. 能解释 logits、sigmoid、CrossEntropy 的角色差异。
4. 能说明 train loss 下降为什么不等于泛化好。
5. 能定位三个 demo 的训练循环、metrics、checkpoint 和 ONNX 检查。
6. 能在 `report.md` 写出一次失败诊断和证据链。
