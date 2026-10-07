# 稀疏比 / 通信 / 吞吐口径推导

## 稀疏比

`sparse_ratio = n_experts / top_k`

| 配置 | 稀疏比 |
|---|---|
| 256 专家 / top_k=8 | 32× |
| 1T / 49B 活跃（Le Chonk 口径） | ≈ 20.4× |

**含义**：每 token 的前向计算量 = 稠密模型的 `top_k / n_experts` 倍。总参数可以堆到 1T，但每步只激活 49B，推理硬件门槛大幅下降。

## 通信随 top_k 走（DDP 式专家并行）

把"搬权重"反转为"搬 token"：专家常驻、token 路由。每步通信量 ∝ `seq_len × top_k × dim`，与专家总数**解耦**。这正是 MoE 能横扩到上千专家的原因。

## 长上下文

滑窗 + 全局 token 下，注意力成本 ∝ `seq × (window + n_global)`，而非 `seq²`。`attention.receptive_field()` 给出平均感受野，可用来做复杂度代理。

> 注意：本项目 `configs/default.yaml` 的可运行 MoE 是**小尺寸**（16 专家）以便 CPU 秒级跑通；1T/49B 是文档化的前沿尺度，由本页公式解析给出，**不要**在纯 Python 里实例化真实 1T MoE。
