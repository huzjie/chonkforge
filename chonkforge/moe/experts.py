"""SwiGLU expert (feed-forward with gated activation)."""
import math


class SwiGLUExpert:
    """y = W2 ( swish(W1 x) * (W3 x) ). W1/W3: dim->hidden, W2: hidden->dim."""

    def __init__(self, dim, hidden, seed=0):
        self.dim = dim
        self.hidden = hidden
        r = __import__("random").Random(seed)
        self.w1 = [[r.uniform(-0.1, 0.1) for _ in range(hidden)] for _ in range(dim)]
        self.w3 = [[r.uniform(-0.1, 0.1) for _ in range(hidden)] for _ in range(dim)]
        self.w2 = [[r.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(hidden)]

    def forward(self, x):
        h1 = [0.0] * self.hidden
        h3 = [0.0] * self.hidden
        for j in range(self.hidden):
            h1[j] = sum(x[k] * self.w1[k][j] for k in range(self.dim))
            h3[j] = sum(x[k] * self.w3[k][j] for k in range(self.dim))
        g = [h1[j] / (1.0 + math.exp(-h1[j])) * h3[j] for j in range(self.hidden)]
        out = [0.0] * self.dim
        for i in range(self.dim):
            out[i] = sum(g[j] * self.w2[j][i] for j in range(self.hidden))
        return out
