# 原生多模态的取舍

## 为什么"native"

- 单一 token 空间 → 单一注意力图 → 文本与视觉天然对齐。
- 无需冻结视觉编码器，视觉 patch 可随主干一起微调。
- 任意分辨率：patch 化天然支持变长图像，无固定分辨率约束。

## 代价

- 视觉 patch 数 = `(H/patch) × (W/patch)`，高分辨率图会撑爆序列长度 → 需要滑窗注意力兜底。
- 早期层要同时学文本与视觉特征，对预训练数据配比更敏感。

## 本项目的落地

`PatchEmbedder(dim, patch, img_h, img_w)` 把 `patch×patch` 块线性投影到 `dim`；`NativeFusion` 用 `<img>...</img>` 特殊 token 包裹视觉 patch id，`segment_ids` 标记模态。
