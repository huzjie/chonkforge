# 配方：CI 接入

1. 提交推送到 main。
2. `.github/workflows/ci.yml` 触发：compileall -> doctor -> unittest。
3. 三档 Python 全绿才合并。
