"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.mm.vision import make_image, PatchEmbedder


class TestVision(unittest.TestCase):
    def test_patch_count(self):
        pe = PatchEmbedder(dim=16, patch=4, img_h=8, img_w=8)
        img = make_image(8, 8)
        tok = pe.embed(img)
        self.assertEqual(len(tok.patch_ids), 4)  # 2x2 grid
        self.assertEqual(len(tok.vectors), 4)
        self.assertEqual(len(tok.vectors[0]), 16)


if __name__ == "__main__":
    unittest.main()
