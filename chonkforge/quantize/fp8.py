"""FP8 (E4M3) quantization simulation.

Reusable idea: FP8 stores 1 sign + 4 exponent + 3 mantissa bits. Quantizing
weights to FP8 (per-tensor or block scale) roughly halves memory vs BF16 with
small accuracy loss. Here we simulate the roundtrip to measure quantization
error, which is what tells you whether FP8 is viable for your model.
"""


class FP8:
    @staticmethod
    def e4m3_max():
        # largest E4M3 magnitude (no inf/nan encodings)
        return 448.0

    @staticmethod
    def quantize(x, scale=1.0):
        """Round a float to the nearest E4M3 representable value."""
        m = FP8.e4m3_max()
        x = max(-m, min(m, x)) * scale
        if x == 0:
            return 0.0
        import math
        sign = -1.0 if x < 0 else 1.0
        a = abs(x)
        e = math.floor(math.log2(a))  # exponent
        # clamp exponent to E4M3 range [-6, 7]
        e = max(-6, min(7, e))
        # 3 mantissa bits -> 8 quantization levels in [2^e, 2^(e+1))
        frac = a / (2.0 ** e) - 1.0  # in [0, 1)
        m_level = round(frac * 8.0) / 8.0
        return sign * (1.0 + m_level) * (2.0 ** e)

    @staticmethod
    def quantize_tensor(vec):
        """Quantize a vector of floats; return (qvec, max_abs, err)."""
        q = [FP8.quantize(v) for v in vec]
        err = sum((a - b) ** 2 for a, b in zip(vec, q)) / max(len(vec), 1)
        return q, max(abs(v) for v in vec), err
