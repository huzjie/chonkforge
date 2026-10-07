# 负载均衡损失

给定一批 token 的路由结果（每个 token 的 top-k 专家与 router 概率），推导负载均衡损失：
- f_e = token 被派到专家 e 的比例
- P_e = 专家 e 的平均 router 概率
- loss = n_experts × Σ f_e · P_e

请说明该损失的物理含义、为什么要乘 n_experts，以及何时需要关闭它。
