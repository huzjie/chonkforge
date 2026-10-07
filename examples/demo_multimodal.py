"""chonkforge example.

Run from anywhere:  python examples/demo_multimodal.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chonkforge.config import load_config
from chonkforge.model.engine import ChonkEngine
from chonkforge.mm.vision import make_image


def main():
    cfg = load_config()
    engine = ChonkEngine(cfg.model)
    print("== native multimodal forward ==")
    img = make_image(cfg.model.img_h, cfg.model.img_w, key="demo")
    tok = engine.patch.embed(img)
    print("image %dx%d -> %d visual patch tokens"
          % (cfg.model.img_h, cfg.model.img_w, len(tok.patch_ids)))
    seg = engine.fusion.build([("text", "what is in this image?"), ("image", tok)])
    print("interleaved sequence length = %d (segments: %d text / %d vision)"
          % (len(seg), seg.segment_ids.count(0), seg.segment_ids.count(1)))
    hidden = engine.transformer.forward(seg.embeddings)
    pooled = engine.transformer.pooled(hidden)
    print("pooled representation[0..3] = %s"
          % [round(x, 4) for x in pooled[:4]])


if __name__ == "__main__":
    main()
