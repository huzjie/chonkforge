"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.mm.tokenizer import TextTokenizer, SPECIAL_TOKENS


class TestTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = TextTokenizer()
        s = "hello world 你好"
        self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_wrap(self):
        tok = TextTokenizer()
        ids = tok.wrap("hi")
        self.assertEqual(ids[0], SPECIAL_TOKENS["<bos>"])
        self.assertEqual(ids[-1], SPECIAL_TOKENS["<eos>"])


if __name__ == "__main__":
    unittest.main()
