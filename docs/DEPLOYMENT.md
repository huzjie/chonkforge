# 部署

## Docker

```bash
docker build -f deploy/Dockerfile -t chonkforge .
docker run -p 8000:8000 chonkforge
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

## Helm

```bash
helm install chonk deploy/helm/chonkforge
```

## CI

`.github/workflows/ci.yml` 在 Python 3.9/3.11/3.13 三档跑 compileall + doctor + 单测。
