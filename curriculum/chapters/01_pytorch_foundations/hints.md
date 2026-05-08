# 01 Problem Set Hints

1. contract 是“必须同时满足的约束”。
2. rank 是维度数，batch dim 通常是第 0 维。
3. parameter 会训练，buffer 通常不训练但随模型保存。
4. eval 控制模块状态，inference_mode 控制梯度记录。
5. export 写文件，check 比输出。
6. `[3,4]` 是二维。
7. Linear 只改最后一维。
8. 每个 hidden layer 把最后一维变成 16。
9. 小心隐式广播。
10. grad 和 parameter 同 shape。
11. 注意 PyTorch weight 是 `[out,in]`。
12. 用 00 章链式法则。
13. optimizer 需要逐元素更新参数。
14. dynamic axes 只标记指定维度。
15. `[32]` 与 `[32,1]` 是常见例子。
16. 搜索 `torch.randn`。
17. 搜索 `x.ndim`。
18. 搜索函数 `export_to_onnx`。
19. 搜索 `max_abs_diff <= atol`。
20. 路径在 `output_path`。
21. 最后一维不等于 input_dim。
22. passed 用 max，不用 mean。
23. 同一算子不能跨设备。
24. CrossEntropyLoss 对 label dtype 很严格。
25. Dropout/BatchNorm 可能行为不稳定。
