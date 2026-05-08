# Learning Readiness Audit

本文件取代旧的 self-study audit。当前仓库不再笼统宣称全仓适合小白独立学习，而是按章节分级。

## 当前判断

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 打开仓库知道怎么学 | `README.md` 指向 `curriculum/README.md` | pass |
| 有分级状态 | `curriculum/status.md` | pass |
| 小白可跟章节 | `00-02` 有 walkthrough、hints、solutions | pass |
| 不误导学习者 | README 明确不默认会 ML/PyTorch/RL，也不宣称全仓 beginner-ready | pass |
| 工程实现稳定 | `src/drl_lab/` 和 tests 保持独立 | pass |
| Spinning Up 来源保留 | `external/spinningup/` | pass |

## 仍需改进

- `03-04` 还只是 usable，需要补完整训练营式解析。
- `05` 是 advanced，不能作为小白入口。
- `07-08` 是工程专题，需要配合算法章节学习。
