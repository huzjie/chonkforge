# 配方：HTTP 服务部署

1. `python -m chonkforge serve --port 8000`。
2. `curl /health`、`POST /v1/chat/completions`。
3. Docker：`docker build -f deploy/Dockerfile .`。
4. K8s：`kubectl apply -f deploy/k8s/` 或 `helm install chonk deploy/helm/chonkforge`。
