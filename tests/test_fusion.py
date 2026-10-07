"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.config import load_config
from chonkforge.model.engine import ChonkEngine
from chonkforge.mm.vision import make_image


class TestFusion(unittest.TestCase):
    def test_interleave(self):
        cfg = load_config()
        engine = ChonkEngine(cfg.model)
        img = make_image(cfg.model.img_h, cfg.model.img_w)
        tok = engine.patch.embed(img)
        seg = engine.fusion.build([("text", "what is this?"), ("image", tok)])
        self.assertGreater(len(seg), 0)
        self.assertTrue(any(s == 1 for s in seg.segment_ids))


if __name__ == "__main__":
    unittest.main()
