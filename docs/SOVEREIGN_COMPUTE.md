# 主权计算与数据治理

Le Chonk 的一个战略点是"欧洲主权"：模型在欧洲数据中心训练，让受监管企业/公共部门对数据驻留、部署连续性、供应链风险有控制权。

`sovereign/` 给出一层**轻量策略抽象**：不搬字节，只**断言并审计**字节允许落在哪。

- `SovereignCompute(region, jurisdiction, data_residency, allowed_regions)`
- `DataGovernance.compute.check(request_region) -> (ok, reason)`
- `AuditLog` 追加式审计（访问/导出事件）

接入真实系统时，把 `check` 换成 KMS 信封加密 + 区域路由断言即可。
