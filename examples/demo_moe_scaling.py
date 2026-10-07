"""chonkforge example.

Run from anywhere:  python examples/demo_moe_scaling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demonstrate how the sparse ratio scales with expert count (analytical, no big alloc)."""


def main():
    for n_experts in (8, 16, 32, 64, 128, 256, 512, 1024):
        top_k = 8
        sparse = n_experts / top_k
        print("n_experts=%4d top_k=%d -> sparse_ratio=%.1fx" % (n_experts, top_k, sparse))
    print("\n1T / 49B active ~= 20.4x  (matches n_experts/top_k ~= 163/8)")


if __name__ == "__main__":
    main()
