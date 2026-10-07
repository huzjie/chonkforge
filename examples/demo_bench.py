"""chonkforge example.

Run from anywhere:  python examples/demo_bench.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chonkforge.config import load_config
from chonkforge.backends import get_backend


def main():
    cfg = load_config()
    backend = get_backend("mock", cfg)
    from chonkforge.bench import run_all
    results = run_all(backend)
    print("%-20s %-10s %s" % ("domain", "accuracy", "n"))
    for name, r in sorted(results.items()):
        print("%-20s %-10.4f %d" % (name, r["accuracy"], r["n"]))
    avg = sum(r["accuracy"] for r in results.values()) / max(len(results), 1)
    print("AVG = %.4f" % avg)


if __name__ == "__main__":
    main()
