"""Configuration loading with a zero-dependency YAML fallback."""
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _load_yaml(path):
    if _HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    from .utils.yamlish import parse
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read()) or {}


@dataclass
class ModelConfig:
    vocab: int = 256
    dim: int = 32
    hidden: int = 64
    n_experts: int = 256
    top_k: int = 8
    n_layers: int = 2
    n_heads: int = 4
    window: int = 32
    n_global: int = 4
    patch: int = 4
    img_h: int = 16
    img_w: int = 16
    total_params_hint: str = "1T"
    active_params_hint: str = "49B"


@dataclass
class BackendConfig:
    name: str = "mock"
    seed: int = 0
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class TrainConfig:
    epochs: int = 4
    lr: float = 0.1
    steps: int = 40


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    @staticmethod
    def from_dict(d):
        m = d.get("model", {}) or {}
        b = d.get("backend", {}) or {}
        t = d.get("train", {}) or {}
        s = d.get("serve", {}) or {}
        return Config(
            model=ModelConfig(**{k: v for k, v in m.items() if k in ModelConfig.__dataclass_fields__}),
            backend=BackendConfig(**{k: v for k, v in b.items() if k in BackendConfig.__dataclass_fields__}),
            train=TrainConfig(**{k: v for k, v in t.items() if k in TrainConfig.__dataclass_fields__}),
            serve=ServeConfig(**{k: v for k, v in s.items() if k in ServeConfig.__dataclass_fields__}),
        )


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    p = Path(path)
    data = {}
    if p.exists():
        data = _load_yaml(p)
    return Config.from_dict(data)
