# 实现思路与踩坑（干货）

## 1. 原生多模态 ≠ 拼接编码器

**反模式**：视觉编码器 + 适配器 + 文本 LLM，三块各自的 token 空间对不齐，适配器成了新的瓶颈。

**做法**：patch 线性投影进文本嵌入的同维空间，视觉 token 与文本 token 交织成**一条**序列喂给同一个 MoE 主干。好处是 attention / MoE 层完全不感知模态差异，`segment_ids` 只作辅助标记。

**踩坑**：投影权重维度必须严格等于 `dim`，否则下游 MoE 的 SwiGLU 专家输入维度对不上会 `IndexError`。

## 2. 稀疏比的口径别虚高

稀疏比 = `n_experts / top_k`，**不是** `total / 每专家参数`。通信量随 `top_k` 走、不随专家总数走，这是 MoE 能线性扩展的前提。负载均衡损失要乘 `n_experts` 归一化，否则梯度量级随专家数漂移。

## 3. 确定性 mock 的两条铁律

① **skill 必须真正参与打分**：`score[i] = skill * (i==correct) + noise`，训练调 `skill` 才有用，否则准确率≈随机。
② **训练信号要单调趋 1**：`delta` 恒正（`skill += lr * (1 - skill)`），别用「对 +1 错 -1」（skill 低时会被压到 0）。

## 4. 确定性随机数的正确姿势

`random.Random(md5(key).digest()[:8] 作为种子)`。直接把 16 字节 digest 平铺取 chunk，只有 ~2 个不同值周期重复，会毁掉噪声的熵。

## 5. 零依赖要自带 YAML 回退

托管 Python 可能没装 `pyyaml`，`config.py` 用 `utils/yamlish.py`（极简 YAML 子集：标量/块列表/行内注释）兜底，否则读 `.yaml` 直接 `ModuleNotFoundError`。
