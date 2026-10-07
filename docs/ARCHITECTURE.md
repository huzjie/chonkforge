# 架构与数据流

```
                      ┌────────────────────────────────────────────┐
  text ──▶ TextTokenizer ──┐                                      │
                           ├─▶ NativeFusion ──▶ ChonkTransformer ──┼─▶ pooled ─▶ score/answer
  image ─▶ PatchEmbedder ──┘   (交织 token)      (滑窗注意力+稀疏MoE) │
                                                                    │
        ┌────────────── 可插拔 backends ──────────────┐             │
        │ mock(可训练) │ cpu │ openai │ vllm │ transformers │       │
        └──────────────────────────────────────────────┘             │
                          │                                        │
                          ▼                                        ▼
                    train / bench / serve / quantize         sovereign(治理)
```

## 分层

1. **core** — 纯 Python 张量 + 注意力。`attention.SlidingWindowGlobalAttention` 是 1M 上下文的核心。
2. **mm** — `tokenizer`（字节级）、`vision`（patch 投影）、`fusion`（交织成单序列）。
3. **moe** — `routing`（top-k + 负载均衡）、`experts`（SwiGLU）、`layer`（路由→分发→专家→加权）。
4. **model** — `transformer`（单主干）、`engine`（高层门面）、`world`（确定性真值）。
5. **backends** — 统一 `answer/train_step` API，注册表在 `backends/__init__.py`。
6. **train / bench / quantize / sovereign / serving / cli** — 上层能力。

## 关键数据流

一次多模态问答：`segments -> NativeFusion.build -> MultimodalSequence -> transformer.forward -> pooled -> score_options -> argmax`。
