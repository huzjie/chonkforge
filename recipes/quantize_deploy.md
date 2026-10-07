# 配方：量化后再部署

1. `python -m chonkforge quantize` 看 FP8/INT4 误差。
2. 误差可接受（<1e-2）→ 用 FP8 压显存减半、INT4 压 4×。
3. 不可接受 → 回 FP8 或增大 group_size。
