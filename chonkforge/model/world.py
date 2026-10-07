"""Deterministic world truth shared by the mock backend and all benchmarks.

A single deterministic function maps (domain, question, n_options) to a correct
option index. Both the mock backend (scoring) and the benchmarks (targets) call
it, so "what the mock knows" always aligns with "what the benchmark grades".
"""
import hashlib


def world_answer(domain, question, n_options):
    seed = int.from_bytes(
        hashlib.md5(("chonk:%s:%s" % (domain, question)).encode("utf-8")).digest()[:8],
        "big",
    )
    return seed % max(n_options, 1)
