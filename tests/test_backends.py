"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.config import load_config
from chonkforge.backends import get_backend, list_backends


class TestBackends(unittest.TestCase):
    def test_registry(self):
        names = list_backends()
        for n in ["mock", "cpu", "openai", "vllm", "transformers"]:
            self.assertIn(n, names)

    def test_mock_answer_valid_index(self):
        b = get_backend("mock", load_config())
        idx = b.answer("coding", "what is O(1)?", ["hash table", "linked list"])
        self.assertIn(idx, (0, 1))

    def test_mock_train_increases_skill(self):
        b = get_backend("mock", load_config())
        s0 = b.get_skill()
        b.train_step("coding", "q", ["A", "B"])
        self.assertGreater(b.get_skill(), s0)

    def test_cpu_runs(self):
        b = get_backend("cpu", load_config())
        idx = b.answer("coding", "what is O(1)?", ["hash table", "linked list"])
        self.assertIn(idx, (0, 1))


if __name__ == "__main__":
    unittest.main()
