"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.moe.layer import MoELayer
from chonkforge.moe.routing import load_balancing_loss


class TestMoE(unittest.TestCase):
    def test_sparse_ratio(self):
        layer = MoELayer(dim=16, hidden=32, n_experts=16, top_k=2, seed=1)
        x = [0.1 * (i % 5) for i in range(16)]
        out, stats = layer.forward(x)
        self.assertEqual(len(out), 16)
        self.assertAlmostEqual(stats.sparse_ratio, 8.0, places=6)

    def test_load_balancing_nonneg(self):
        layer = MoELayer(dim=16, hidden=32, n_experts=8, top_k=2, seed=1)
        x = [0.1 * (i % 5) for i in range(16)]
        routes = [layer.router.route(x) for _ in range(5)]
        loss = load_balancing_loss(routes)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
