# Debug Checklist

这份清单用于每次训练失败、指标异常或导出不一致时排查。先查基础工程问题，再怀疑算法。

## 训练前

- global seed 是否设置。
- 环境 seed 和 action-space seed 是否在支持时设置。
- observation/action shape 是否打印或 assert。
- tensor dtype 是否正确。
- model、input、target 是否在同一 device。
- optimizer 是否包含所有可训练参数。
- reward scale 是否合理。
- `terminated` 和 `truncated` 是否被有意识地处理。

## 训练中

- loss 是否 finite。
- gradient 是否 finite。
- gradient norm 是否在合理范围。
- replay buffer 是否在填充多样样本。
- target network 是否按预期频率更新。
- exploration 是否衰减太快。
- policy-gradient 方法的 entropy 是否过早塌缩。
- value loss 是否压过 policy loss。

## 训练后

- eval 是否关闭探索噪声或 epsilon。
- eval 是否使用 `model.eval()` 和 `torch.inference_mode()`。
- checkpoint 是否能重新加载。
- 同一 checkpoint 是否给出稳定 eval 结果。
- `eval_result.json` 是否记录最终 eval。
- `export_report.json` 是否记录导出路径、输入输出名、shape 和误差。
- ONNXRuntime 是否在容差内匹配 PyTorch。
- 成功或失败实验是否写入 report。
