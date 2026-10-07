"""MoE layer: route -> dispatch -> experts -> weighted combine."""
from dataclasses import dataclass

from .routing import TopKRouter
from .experts import SwiGLUExpert


@dataclass
class SparseStats:
    n_experts: int
    top_k: int
    activated: int
    sparse_ratio: float
    aux_loss: float


class MoELayer:
    def __init__(self, dim, hidden, n_experts, top_k, seed=0):
        self.dim = dim
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router = TopKRouter(dim, n_experts, top_k, seed=seed)
        self.experts = [SwiGLUExpert(dim, hidden, seed=seed + 1000 + i)
                        for i in range(n_experts)]

    def forward(self, x):
        """x: list[float]. Returns (out, sparse_stats)."""
        top, weights, probs = self.router.route(x)
        out = [0.0] * self.dim
        for eid, w in zip(top, weights):
            e_out = self.experts[eid].forward(x)
            for i in range(self.dim):
                out[i] += w * e_out[i]
        aux = 0.0
        # single-token load-balancing contribution (f_e = 1/top_k for selected)
        for eid in top:
            aux += (1.0 / self.top_k) * probs[eid]
        aux *= self.n_experts
        stats = SparseStats(
            n_experts=self.n_experts, top_k=self.top_k,
            activated=self.top_k,
            sparse_ratio=self.n_experts / self.top_k,
            aux_loss=aux,
        )
        return out, stats
