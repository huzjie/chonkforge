"""Top-k expert routing with load-balancing auxiliary loss.

The reusable idea behind extreme sparse MoE: route each token to a *fixed small*
number of experts (top-k), so the compute cost per token scales with k, not with
the total expert count. The load-balancing auxiliary loss nudges the router so
tokens spread across experts (otherwise a few experts dominate and capacity is
wasted).
"""
from ..core.nn import Linear, softmax


class TopKRouter:
    def __init__(self, dim, n_experts, top_k=8, seed=0):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.gate = Linear(dim, n_experts, seed=seed)

    def route(self, x):
        """x: list[float]. Returns (top_expert_ids, gate_weights, router_probs)."""
        logits = self.gate(x)
        probs = softmax(logits)
        ranked = sorted(range(self.n_experts), key=lambda i: logits[i], reverse=True)
        top = ranked[: self.top_k]
        # renormalize gate weights over the selected top-k
        s = sum(probs[i] for i in top) or 1e-9
        weights = [probs[i] / s for i in top]
        return top, weights, probs


def load_balancing_loss(routes):
    """routes: list of (top_ids, weights, probs) for a batch of tokens.

    Returns scalar aux loss = sum_e (f_e * P_e) where f_e is the fraction of
    tokens dispatched to expert e and P_e is the mean router probability for e.
    Lower => more uniform routing.
    """
    n_experts = len(routes[0][2])
    f = [0.0] * n_experts
    P = [0.0] * n_experts
    for top, weights, probs in routes:
        for e in top:
            f[e] += 1.0 / len(routes)
        for e in range(n_experts):
            P[e] += probs[e] / len(routes)
    return sum(f[e] * P[e] for e in range(n_experts)) * n_experts
