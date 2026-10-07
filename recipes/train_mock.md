# 配方：训练 mock 后端

1. `backend = get_backend("mock", cfg)`。
2. `run_pipeline(backend, pretrain_steps=20, sft_steps=30, rl_steps=30)`。
3. 观察 skill 0.5 -> 1.0，三阶段指标单调变化。
