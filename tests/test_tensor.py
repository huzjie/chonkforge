"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.core.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_vector(self):
        t = Tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.shape, (3, 0))
        self.assertTrue(t.is_vector)

    def test_matrix_add(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[10.0, 20.0], [30.0, 40.0]])
        c = a + b
        self.assertEqual(c.data, [[11.0, 22.0], [33.0, 44.0]])

    def test_softmax_sums_to_one(self):
        t = Tensor([1.0, 2.0, 3.0]).softmax()
        self.assertAlmostEqual(sum(t.data), 1.0, places=6)

    def test_matmul(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.matmul(b)
        self.assertEqual(c.data, [[19.0, 22.0], [43.0, 50.0]])

    def test_relu(self):
        t = Tensor([-1.0, 0.0, 2.0]).relu()
        self.assertEqual(t.data, [0.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
