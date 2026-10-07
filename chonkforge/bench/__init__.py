"""Multi-domain benchmark suite + registry."""
_BENCHMARKS = {}


def register_benchmark(name):
    def deco(cls):
        _BENCHMARKS[name] = cls
        return cls
    return deco


def list_benchmarks():
    return sorted(_BENCHMARKS.keys())


from . import coding, agentic, cybersecurity, finance, law, visual_grounding  # noqa: E402,F401
from .runner import run_all  # noqa: E402

__all__ = ["register_benchmark", "list_benchmarks", "run_all"]
