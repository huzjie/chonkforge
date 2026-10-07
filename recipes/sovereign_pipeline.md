# 配方：主权计算接入

1. `sc = SovereignCompute(region="eu", jurisdiction="GDPR", allowed_regions=["eu"])`。
2. `gov = DataGovernance(sc)`。
3. 每次访问/导出走 `gov.access(actor, region, purpose)` / `gov.export(...)`。
4. 接生产时把 `check` 换成 KMS + 区域路由断言。
