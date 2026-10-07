# chonkforge —— 把「原生多模态 1T 稀疏 MoE」拆开给你看

> 今日选题：Mistral Large 4 "Le Chonk"（2026-10-07 发布）。总参数 ~1T、每步仅激活 49B、1M 上下文、欧洲主权算力训练，主打 coding / 智能体 / 安全 / 金融 / 法律 / 视觉定位。
> 本篇不写"它有多强"的资讯，只讲**它背后的技术怎么实现、能抄走哪些思路**。

## 一句话讲清三个技术点

1. **原生多模态**：不是"视觉编码器 + 适配器 + LLM"三块拼装，而是把图像 patch 线性投影进文本嵌入的**同维空间**，文本 token 与视觉 token 交织成**一条序列**，走同一个 MoE 主干——没有第二个编码器。
2. **极致稀疏**：1T 总参 / 49B 活跃 ≈ 20×。关键是每 token 只路由到固定 `top_k` 个专家，计算与通信只随 `top_k` 走、与专家总数解耦，这就是它能横扩到上千专家的原因。
3. **1M 上下文**：滑窗局部注意力 + 少量全局 token。token i 只看 `[i-window+1, i]` + 前 `n_global` 个全局 token；全局 token 看所有人，长程信息经它中转，成本从 O(n²) 降到 O(n·(window+global))。

## 可直接抄走的干货块

**稀疏 MoE 的负载均衡损失**（为什么必须乘 n_experts 归一化）：

```python
def load_balancing_loss(routes):
    n_experts = len(routes[0][2])
    f = [0.0]*n_experts; P = [0.0]*n_experts
    for top, weights, probs in routes:
        for e in top:      f[e] += 1.0/len(routes)
        for e in range(n_experts): P[e] += probs[e]/len(routes)
    return sum(f[e]*P[e] for e in range(n_experts)) * n_experts
```

**可训练 mock 的两条铁律**（训练曲线不翻车的底线）：
- skill 必须真正参与打分：`score[i] = skill * (i==correct) + noise`，否则调它没用、准确率≈随机。
- 训练信号要单调趋 1：`skill += lr*(1-skill)`，别用"对+1 错-1"（skill 低时会被压到 0）。

**确定性随机数**：`random.Random(md5(key).digest()[:8])`。把 16 字节 digest 直接平铺只有 ~2 个值周期重复，噪声熵会塌掉。

## 怎么跑（零依赖，Python 3.9+）

```bash
python -m chonkforge doctor     # 自检
python -m chonkforge train      # skill 0.5 -> 1.0
python -m chonkforge bench      # 六域基准
python -m chonkforge serve      # OpenAI 兼容 HTTP
```

## 学到了什么（可复用清单）

- 原生多模态的**单一 token 空间**，让下游层完全不用感知"这是字还是像素"。
- 稀疏比的正确口径是 `n_experts/top_k`，不是"总参/专家参"。
- 主权计算的本质是一层**断言 + 审计**，不搬字节。
