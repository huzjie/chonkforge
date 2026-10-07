"""A reference natively-multimodal sparse transformer.

One backbone: embeddings -> sliding-window+global attention -> sparse MoE FFN
(repeated). Kept small so it can be instantiated and run on a CPU; the sparse
ratio and routing statistics remain the interesting part.
"""
from ..core.nn import Embedding
from ..core.attention import SlidingWindowGlobalAttention
from ..moe.layer import MoELayer


class ChonkTransformer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = cfg.dim
        self.emb = Embedding(cfg.vocab, cfg.dim, seed=0)
        self.attn = SlidingWindowGlobalAttention(cfg.dim, cfg.window, cfg.n_global)
        self.layers = [MoELayer(cfg.dim, cfg.hidden, cfg.n_experts, cfg.top_k, seed=7 + i)
                       for i in range(cfg.n_layers)]

    def forward(self, embeddings):
        """embeddings: seq x dim (list of rows). Returns hidden seq."""
        x = embeddings
        for layer in self.layers:
            ctx = self.attn(x)
            nxt = []
            for i, row in enumerate(x):
                h = [row[j] + ctx[i][j] for j in range(len(row))]
                h2, _ = layer.forward(h)
                nxt.append([h[j] + h2[j] for j in range(len(h))])
            x = nxt
        return x

    def pooled(self, x):
        dim = len(x[0]) if x else 0
        p = [0.0] * dim
        n = max(len(x), 1)
        for row in x:
            for j in range(dim):
                p[j] += row[j] / n
        return p


def make_transformer(cfg):
    return ChonkTransformer(cfg)
