# FAQ

**Q: 真的零依赖吗？** 核心 + mock/cpu 后端零依赖。openai/vllm/transformers 后端按需 `requests`/`vllm`/`transformers`。

**Q: 为什么 MoE 这么小？** 纯 Python 里实例化 1T 参数会卡死。稀疏比由公式解析给出，见 `docs/SCALING.md`。

**Q: mock 后端准确率为什么能到 1.0？** `skill` 单调趋 1 后，正确选项得分恒高于噪声，是有意设计，用来演示训练曲线。

**Q: 换真实模型怎么做？** 实现 `BaseBackend.answer/train_step` 接你自己的推理即可，管线不变。
