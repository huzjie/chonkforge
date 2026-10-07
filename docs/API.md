# API 速览

## 后端统一接口

```python
from chonkforge.backends import get_backend, list_backends
backend = get_backend("mock", cfg)
backend.answer(domain, question, options, image=None) -> int   # 选项下标
backend.train_step(domain, question, options, image=None) -> {"skill", "correct"}
backend.get_skill() -> float
```

## HTTP（OpenAI 兼容子集）

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/answer`

可选 Bearer 鉴权：`cfg.serve.auth_token`。
