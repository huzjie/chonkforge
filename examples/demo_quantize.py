"""chonkforge example.

Run from anywhere:  python examples/demo_quantize.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chonkforge.utils.stable import stable_vector


def main():
    from chonkforge.quantize import FP8, INT4
    vec = stable_vector("q.demo", 256, -3.0, 3.0)
    _, fp8_amax, fp8_err = FP8.quantize_tensor(vec)
    int4_out, int4_err = INT4.roundtrip(vec, group_size=32)
    print("FP8  amax=%.4f mse=%.6f" % (fp8_amax, fp8_err))
    print("INT4 mse=%.6f" % int4_err)


if __name__ == "__main__":
    main()
