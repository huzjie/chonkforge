# 配方：基准进 CI

1. 在 CI 加 `python -m chonkforge bench`。
2. 设 AVG 阈值断言（如 > 0.6）。
3. 训练前后各跑一次，输出提升幅度。
