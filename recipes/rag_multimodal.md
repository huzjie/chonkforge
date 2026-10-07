# 配方：多模态 RAG

1. 图像 patch 化 -> 视觉 token。
2. 视觉 token + 文本 token 建索引。
3. 检索时用 `pooled` 表示做相似度。
4. 用 `prompts/rag.md` 约束模型只依据检索片段作答。
