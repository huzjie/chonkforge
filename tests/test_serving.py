"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.config import load_config
from chonkforge.backends import get_backend


class TestServing(unittest.TestCase):
    def test_chat_completion(self):
        b = get_backend("mock", load_config())
        from chonkforge.serving import make_server
        app = make_server(b)
        body = {"messages": [{"role": "user", "content": "hi"}]}
        resp = app.chat_completion(body)
        self.assertIn("choices", resp)
        self.assertEqual(len(resp["choices"]), 1)

    def test_answer(self):
        b = get_backend("mock", load_config())
        from chonkforge.serving import make_server
        app = make_server(b)
        resp = app.answer({"domain": "coding", "question": "q",
                           "options": ["A", "B"]})
        self.assertIn(resp["answer"], (0, 1))


if __name__ == "__main__":
    unittest.main()
