# 07 Experiment Engineering Lab Guide

## 前置阅读

先读 `curriculum/chapters/07_experiment_engineering/lesson.md`、`docs/debug_checklist.md` 和 `docs/experiment_report_template.md`。

## 实验目标

- 跑通 snapshot demo。
- 检查 config、environment、metrics、checkpoint 的职责。
- 学会写失败报告，而不是只记录成功结果。

## 代码入口

```bash
conda run -n drl-lab python curriculum/chapters/07_experiment_engineering/code/snapshot_demo.py
```

工程实现：`src/drl_lab/common/experiment.py`。测试：`tests/test_experiment_snapshots.py`。

## 提交产物

- 完成 `exercises.md`。
- 填写 `report.md`。
- 说明一次训练 run 应该保存哪些文件。

## 常见坑

- 只保存模型，不保存 config。
- eval 和 train 混用探索策略。
- 指标文件无法和 checkpoint 对齐。
