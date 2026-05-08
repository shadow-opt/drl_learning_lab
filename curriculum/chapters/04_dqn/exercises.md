# DQN Exercises

## 纸笔题

1. 从 tabular Q-learning 写出 DQN TD target。
2. 解释 replay buffer 为什么能降低样本相关性。
3. 解释 target network 为什么能稳定训练。
4. 说明 Q-network 的 ONNX 输入输出分别是什么。

## 代码题

1. 跑通 DQN smoke demo。
2. 跑一个短 CartPole 训练。
3. 打开 `src/drl_lab/algorithms/dqn/losses.py`，指出 TD target。
4. 打开 `tests/test_dqn.py`，说明测试覆盖了哪些核心组件。

## 观察题

1. epsilon 是否随训练变化？
2. episode return 是否有改善迹象？
3. checkpoint 和 ONNX artifact 是否生成？
4. `eval_result.json` 和 `export_report.json` 是否能说明最终 eval 与 ONNX 误差？

## 提交物

- 填完 `report.md`。
- 写出 DQN target。
- 记录一次训练或导出观察。
