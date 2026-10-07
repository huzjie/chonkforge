# Changelog

## 1.0.0 (2026-10-07)

- 原生多模态融合（单一 token 空间 + 单一 MoE 主干）
- 极致稀疏 MoE（top-k 路由 + SwiGLU 专家 + 负载均衡损失）
- 1M 上下文滑窗 + 全局 token 注意力
- 六域基准 + 确定性可训练 mock 后端
- FP8/INT4 量化 + 主权计算治理
- stdlib HTTP 服务（OpenAI 兼容）+ CLI + Docker/K8s/Helm/CI
