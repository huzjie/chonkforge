"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.utils.stable import stable_float, stable_ints


class TestStable(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(stable_float("k"), stable_float("k"))

    def test_range(self):
        for _ in range(20):
            v = stable_float("x%d" % _)
            self.assertTrue(0.0 <= v <= 1.0)

    def test_ints(self):
        vals = stable_ints("k", 100, 0, 9)
        self.assertEqual(len(vals), 100)
        self.assertTrue(all(0 <= v <= 9 for v in vals))


if __name__ == "__main__":
    unittest.main()
