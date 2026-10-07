"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.config import load_config
from chonkforge.backends import get_backend


class TestBench(unittest.TestCase):
    def test_run_all(self):
        b = get_backend("mock", load_config())
        from chonkforge.bench import run_all
        results = run_all(b)
        self.assertEqual(len(results), 6)
        for r in results.values():
            self.assertIn("accuracy", r)


if __name__ == "__main__":
    unittest.main()
