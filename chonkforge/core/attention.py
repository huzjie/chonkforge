"""1M-context attention: interleaved sliding-window + global tokens.

The reusable idea behind the 1M-token context: make attention cost grow with the
window, not the full sequence. Every token attends to (a) its local sliding
window, and (b) a tiny set of "global" tokens that themselves attend everywhere
-- so long-range information propagates through the global tokens in O(seq *
(window + n_global)) instead of O(seq^2).
"""
from .nn import scaled_dot_product


def causal_mask(seq, window, n_global):
    """Build a boolean attention mask: token i attends to j if
    (j <= i) and (j >= i - window + 1 or j < n_global)."""
    m = []
    for i in range(seq):
        row = []
        for j in range(seq):
            if j > i:
                row.append(False)
            elif j < n_global:
                row.append(True)
            elif j >= i - window + 1:
                row.append(True)
            else:
                row.append(False)
        m.append(row)
    return m


class SlidingWindowGlobalAttention:
    """Sliding-window local attention augmented with global tokens."""

    def __init__(self, dim, window=32, n_global=4):
        self.dim = dim
        self.window = window
        self.n_global = n_global

    def __call__(self, x):
        # x: seq x dim (list of rows). Uses identity q=k=v for the reference impl.
        seq = len(x)
        mask = causal_mask(seq, self.window, self.n_global)
        return scaled_dot_product(x, x, x, mask=mask)

    @staticmethod
    def receptive_field(seq, window, n_global):
        """Average number of positions each token attends to (a complexity proxy)."""
        total = 0
        for i in range(seq):
            total += min(i + 1, window) + min(n_global, i + 1)
        return total / max(seq, 1)
