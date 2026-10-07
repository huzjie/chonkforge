# 快速上手

1. **环境**：Python 3.9+，无需安装任何依赖（核心 + mock/cpu 后端零依赖）。
2. **自检**：`python -m chonkforge doctor`
3. **训练**：`python -m chonkforge train`（skill 0.5 -> 1.0）
4. **评测**：`python -m chonkforge bench`
5. **服务**：`python -m chonkforge serve`，然后 `curl 127.0.0.1:8000/health`
6. **量化**：`python -m chonkforge quantize`

## 换真实后端

在 `configs/default.yaml` 改 `backend.name`，或 CLI `--backend openai`，并填 `base_url` / `api_key` / `model`。
