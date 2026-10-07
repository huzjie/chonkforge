"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.config import load_config
from chonkforge.backends import get_backend


class TestTrain(unittest.TestCase):
    def test_pipeline_monotonic(self):
        b = get_backend("mock", load_config())
        from chonkforge.train import run_pipeline
        res = run_pipeline(b, pretrain_steps=15, sft_steps=15, rl_steps=15)
        self.assertAlmostEqual(b.get_skill(), 1.0, places=1)
        # pretrain loss decreased
        self.assertLess(res["pretrain"][-1][2], res["pretrain"][0][2])


if __name__ == "__main__":
    unittest.main()
