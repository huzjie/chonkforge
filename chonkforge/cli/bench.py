"""bench: run the 6-domain benchmark suite."""
from ..backends import get_backend


def bench(cfg):
    backend = get_backend(cfg.backend.name, cfg)
    from ..bench import run_all
    results = run_all(backend)
    print("\n%-20s %-10s %s" % ("domain", "accuracy", "n"))
    print("-" * 40)
    for name, r in sorted(results.items()):
        print("%-20s %-10.4f %d" % (name, r["accuracy"], r["n"]))
    avg = sum(r["accuracy"] for r in results.values()) / max(len(results), 1)
    print("-" * 40)
    print("%-20s %-10.4f" % ("AVG", avg))
