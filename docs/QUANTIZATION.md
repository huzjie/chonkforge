# FP8 / INT4 的工程选择

## FP8 (E4M3)

1 符号 + 4 指数 + 3 尾数。指数范围 [-6, 7]，无 inf/nan 编码。量化误差比 BF16 大，但显存减半。适合**权重 + 激活**同时压。

## INT4 (group-wise absmax)

把权重按 `group_size` 分组，每组用 abs-max 缩放到位格 [-8, 7]。group 越小组内误差越小、但 scale 开销越大（`group_size=32` 是常见折中）。适合**权重**量化。

## 判据

`quantize/fp8.py` / `int4.py` 输出均方误差。误差量级直接决定你的部署精度预算——误差 >1e-2 时建议回到 FP8 或增大 group。
