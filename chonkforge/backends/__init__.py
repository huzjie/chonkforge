"""Backend registry. A backend exposes answer/train_step with a uniform API."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        cls.backend_name = name
        return cls
    return deco


def list_backends():
    return sorted(_BACKENDS.keys())


def get_backend(name, cfg=None, **kw):
    if name not in _BACKENDS:
        raise ValueError("unknown backend %r (available=%r)" % (name, list_backends()))
    return _BACKENDS[name](cfg=cfg, **kw)


from . import base, mock, cpu, openai, vllm, transformers  # noqa: E402,F401

__all__ = ["register_backend", "list_backends", "get_backend"]
