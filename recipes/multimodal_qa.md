# 配方：多模态问答

1. 加载配置 `cfg = load_config()`。
2. 建引擎 `engine = ChonkEngine(cfg.model)`。
3. 生成图像 `img = make_image(8, 8)`，patch 化 `tok = engine.patch.embed(img)`。
4. 建序列 `seq = engine.fusion.build([("text", q), ("image", tok)])`。
5. 前向 + pooled + 打分。
