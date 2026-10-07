# chonkforge

> **原生多模态 1T 稀疏 MoE 前沿大模型框架**（Mistral Large 4 / "Le Chonk" 方向）
> 一个零依赖、填好配置即可完整跑通的参考实现：文本 token 与图像 patch token 共享**同一个**嵌入空间与**同一条** MoE 主干，用 1M 上下文的滑窗+全局注意力做长程建模，以 ~20× 稀疏比把总参数推到 1T 而每步只激活 49B，并配套六域能力头、量化部署与主权计算治理。

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Zero-Dependency](https://img.shields.io/badge/core-zero--dependency-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Scale](https://img.shields.io/badge/scale-1T--total--49B--active-red)

---

## 这不是什么

不是一个"今天发生了什么"的资讯项目，也不是某个公司的官方实现。它是一个**把「原生多模态 + 1T 级稀疏 MoE」这套工程思路拆开、做成可运行代码**的技术框架。读完你能照着把下面三件事在自己的项目里复现出来。

## 三个可复用的核心技术

### 1. 原生多模态：文本和图像走「同一条主干」

大多数多模态模型 = 一个**冻结的视觉编码器** + 一个适配器 + 一个文本 LLM，三块拼在一起。Le Chonk 的"native"是把图像 patch 直接线性投影进文本 token 的嵌入空间，然后**文本 token 和视觉 token 交织成一条序列**，喂给同一个 MoE transformer——没有第二个编码器，没有 cross-attention 桥。

```python
from chonkforge.mm.vision import PatchEmbedder, make_image
from chonkforge.mm.fusion import NativeFusion
from chonkforge.model.engine import ChonkEngine

engine = ChonkEngine(cfg.model)
img = make_image(8, 8)                    # 一张 8x8 的合成图
tok = engine.patch.embed(img)             # -> 4 个视觉 patch token（与文本同维）
seq = engine.fusion.build([("text", "图里是什么？"), ("image", tok)])
# seq.token_ids:   [<bos>] + 文本字节 + <img> + 4个视觉id + </img> + [<eos>]
# seq.segment_ids: 0(文本) / 1(视觉) 标记，用于模态感知
```

**工程要点**：视觉 patch 的投影权重 `PatchEmbedder.proj` 与文本嵌入表共享一个 `dim`，这样下游的 attention / MoE 层**完全不用知道**某个 token 是字还是像素。

### 2. 极致稀疏 MoE：计算只随 top-k 走，不随专家总数走

1T 参数 / 每步 49B 活跃 ≈ 20× 稀疏比。关键不是"专家多"，而是**每 token 只路由到固定个数的专家**，于是计算量、通信量都只跟 `top_k` 相关，与专家总数解耦。

```python
from chonkforge.moe.layer import MoELayer
layer = MoELayer(dim=16, hidden=32, n_experts=256, top_k=8)
_, stats = layer.forward(x)
# stats.sparse_ratio == 256 / 8 == 32x   <- 专家越多、top_k 越小，稀疏比越高
```

配套一个**负载均衡辅助损失** `load_balancing_loss()`，让 token 在专家间铺开，避免少数专家过载、其余闲置。详细推导见 `docs/SCALING.md`。

### 3. 1M 上下文：滑窗 + 全局 token，把 O(n²) 压成 O(n·(window+global))

```python
from chonkforge.core.attention import SlidingWindowGlobalAttention, causal_mask
attn = SlidingWindowGlobalAttention(dim, window=32, n_global=4)
# token i 只看 [i-window+1, i] 的局部 + 前 n_global 个全局 token
# 全局 token 看所有人 -> 长程信息经全局 token 中转，成本近似线性
```

## 快速开始

```bash
# 零依赖，Python 3.9+ 直接跑
python -m chonkforge doctor        # 环境 + 组件自检
python -m chonkforge train         # pretrain -> sft -> rl（skill 0.5 -> 1.0）
python -m chonkforge bench         # 六域基准（coding/agentic/cyber/finance/law/visual）
python -m chonkforge quantize      # FP8 / INT4 量化误差
python -m chonkforge serve         # 起 stdlib HTTP 服务（OpenAI 兼容子集）
```

六域基准（确定性可训练 mock 后端，`skill` 从 0.5 单调升到 1.0）：

| 域 | 内容 | 模态 |
|---|---|---|
| coding | 算法/数据结构/工程 | 文本 |
| agentic | 智能体规划/工具/护栏 | 文本 |
| cybersecurity | 注入/XSS/零信任 | 文本 |
| finance | 复利/估值/对冲 | 文本 |
| law | 合同/知识产权/合规 | 文本 |
| visual_grounding | 空间/目标/颜色 | 文本+图像 |

## 目录结构

```
chonkforge/
  core/        纯 Python 张量 + 注意力（滑窗+全局）
  mm/          原生多模态：分词 / patch 视觉编码 / 融合
  moe/         top-k 路由 / SwiGLU 专家 / 负载均衡
  model/       transformer + engine + 确定性 world truth
  backends/    mock(可训练)/cpu/openai/vllm/transformers
  train/       pretrain / sft / grpo-ppo
  quantize/    FP8(E4M3) / INT4(group-wise absmax)
  sovereign/   主权计算 + 数据治理审计
  bench/       六域基准 + runner
  serving/     stdlib HTTP（OpenAI 兼容）
  cli/         doctor/train/bench/serve/quantize
examples/      可运行示例
docs/          ARCHITECTURE / TECH_INSIGHTS / SCALING / ...
prompts/       提示词库        recipes/  配方
```

## 文档

- `docs/ARCHITECTURE.md` — 整体架构与数据流
- `docs/TECH_INSIGHTS.md` — 实现思路与踩坑（干货）
- `docs/SCALING.md` — 稀疏比 / 通信 / 吞吐口径推导
- `docs/MULTIMODAL.md` — 原生多模态的取舍
- `docs/TRAINING.md` — 为什么用 `skill` 做能力旋钮
- `docs/QUANTIZATION.md` — FP8/INT4 的工程选择
- `prompts/` — 12 个可直接复用的提示词模板
- `recipes/` — 6 个端到端配方

## License

MIT。这是对 Mistral Large 4 公开工程思路的**独立参考实现**，与 Mistral AI 无隶属关系。
