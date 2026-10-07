"""chonkforge example.

Run from anywhere:  python examples/demo_train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chonkforge.config import load_config
from chonkforge.backends import get_backend


def main():
    cfg = load_config()
    backend = get_backend("mock", cfg)
    print("== trainable mock: skill 0.5 -> 1.0 ==")
    from chonkforge.train import run_pipeline
    res = run_pipeline(backend, pretrain_steps=20, sft_steps=30, rl_steps=30)
    print("pretrain final loss = %.4f" % res["pretrain"][-1][2])
    print("sft final accuracy  = %.4f" % res["sft"][-1][2])
    print("rl final reward     = %.4f" % res["rl"][-1][2])
    print("final skill = %.3f" % backend.get_skill())


if __name__ == "__main__":
    main()
