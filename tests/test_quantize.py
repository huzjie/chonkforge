"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.utils.stable import stable_vector


class TestQuantize(unittest.TestCase):
    def test_fp8(self):
        from chonkforge.quantize import FP8
        vec = stable_vector("q.fp8", 64, -2.0, 2.0)
        q, amax, err = FP8.quantize_tensor(vec)
        self.assertEqual(len(q), 64)
        self.assertGreaterEqual(err, 0.0)

    def test_int4(self):
        from chonkforge.quantize import INT4
        vec = stable_vector("q.int4", 64, -2.0, 2.0)
        out, err = INT4.roundtrip(vec, group_size=16)
        self.assertEqual(len(out), 64)
        self.assertGreaterEqual(err, 0.0)


if __name__ == "__main__":
    unittest.main()
