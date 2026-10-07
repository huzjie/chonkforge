"""Benchmark runner: dispatch per-benchmark (each may take different kwargs)."""
from ..utils.logging import get_logger
from . import _BENCHMARKS


def run_all(backend, domains=None, n_tasks=None):
    """Run every registered benchmark. Returns {domain: summary}."""
    log = get_logger("bench")
    results = {}
    for name in sorted(_BENCHMARKS.keys()):
        if domains and name not in domains:
            continue
        b = _BENCHMARKS[name]()
        if n_tasks is not None:
            b.n_tasks = n_tasks
        res = b.run(backend)
        results[name] = res
        log.info("bench[%s] accuracy=%.3f (%d/%d)",
                 name, res["accuracy"], res["correct"], res["n"])
    if results:
        avg = sum(r["accuracy"] for r in results.values()) / len(results)
        log.info("bench AVG = %.4f across %d domains", avg, len(results))
    return results
