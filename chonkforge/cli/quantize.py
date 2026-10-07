"""quantize: FP8/INT4 quantization demo."""
from ..utils.stable import stable_vector


def quantize(cfg):
    from ..quantize import FP8, INT4
    vec = stable_vector("quantize.demo", 128, -2.0, 2.0)

    fp8_q, fp8_amax, fp8_err = FP8.quantize_tensor(vec)
    int4_out, int4_err = INT4.roundtrip(vec, group_size=32)

    print("== FP8 (E4M3) ==")
    print("  amax=%.4f  mean_sq_err=%.6f" % (fp8_amax, fp8_err))
    print("  bits/elem = 8  (vs bf16: 16) -> ~2x memory cut")
    print("== INT4 (group-wise absmax) ==")
    print("  mean_sq_err=%.6f" % int4_err)
    print("  bits/elem = 4  (vs bf16: 16) -> ~4x memory cut")
    print("\nNOTE: quantization error tells you the deployment precision budget.")
