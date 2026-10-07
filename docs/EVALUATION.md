# 评测方法论

## 口径

- **准确率**：`run_all(backend)` 逐域计算，`target = world_answer(domain, question, n)`。
- **训练前 vs 训练后**：skill=0.5 时 AVG ≈ 0.67，训练到 1.0 时 AVG = 1.00（见 `examples/demo_bench.py`）。
- **量化误差**：FP8/INT4 的均方误差是部署精度预算的判据。

## 反直觉点

- 确定性 mock 的准确率随 skill 单调上升，是**有意设计**（能力旋钮），不是过拟合。
- 六域基准的 `target` 与 mock 的 `correct` 共用同一个 `world_answer`，保证"模型知道的"与"评测打分的"对齐。
