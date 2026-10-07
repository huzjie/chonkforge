# 六域基准

每个域 12–16 个确定性任务，`target = world_answer(domain, question, n)`，mock 后端用 `skill * is_correct + noise` 打分，因此准确率随 `skill` 从 ~1/N 单调升到 1.0。

| 域 | 任务数 | 模态 | 说明 |
|---|---|---|---|
| coding | 16 | 文本 | 算法/数据结构/工程 |
| agentic | 16 | 文本 | 规划/工具/护栏 |
| cybersecurity | 16 | 文本 | 注入/XSS/零信任 |
| finance | 16 | 文本 | 复利/估值/对冲 |
| law | 16 | 文本 | 合同/知产/合规 |
| visual_grounding | 12 | 文本+图像 | 空间/目标/颜色 |

`run_all(backend)` 逐域分发（各域参数不同，**不要**把同一 kwargs 塞给所有基准）。
