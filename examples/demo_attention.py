"""chonkforge example.

Run from anywhere:  python examples/demo_attention.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Show the O(n^2) -> O(n*(window+global)) complexity reduction."""
from chonkforge.core.attention import SlidingWindowGlobalAttention


def main():
    attn = SlidingWindowGlobalAttention(dim=16, window=32, n_global=4)
    for seq in (100, 1000, 10000, 100000, 1000000):
        full = seq * seq
        sparse = seq * (32 + 4)
        print("seq=%-8d full=%-12d sparse=%-10d speedup=%.0fx"
              % (seq, full, sparse, full / max(sparse, 1)))


if __name__ == "__main__":
    main()
