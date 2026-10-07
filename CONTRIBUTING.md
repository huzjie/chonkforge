# Contributing

1. 保持核心零依赖（`chonkforge/core`、`mm`、`moe`、`model` 只用标准库）。
2. 新后端继承 `backends/base.py`，用 `@register_backend` 并在 `backends/__init__.py` 显式 import。
3. 新基准继承 `bench/base.py`，用 `@register_benchmark` 并在 `bench/__init__.py` 显式 import。
4. 训练信号保持单调趋 1（恒正 delta）。
5. 提交前跑 `python -m compileall -q chonkforge` 与 `python -m unittest discover -s tests -t .`。
