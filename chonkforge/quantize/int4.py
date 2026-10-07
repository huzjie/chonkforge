"""INT4 group-wise absmax quantization simulation.

Reusable idea: split weights into groups, scale each group by its abs-max to a
4-bit signed integer grid (-8..7). Dequantization multiplies back. Group-wise
scaling is the standard way to keep INT4 accurate (vs a single global scale).
"""


class INT4:
    @staticmethod
    def quantize_group(group):
        """group: list[float]. Returns (qints, scale)."""
        amax = max(abs(v) for v in group) or 1e-9
        scale = amax / 7.0
        q = [max(-8, min(7, round(v / scale))) for v in group]
        return q, scale

    @staticmethod
    def dequantize(qints, scale):
        return [q * scale for q in qints]

    @staticmethod
    def roundtrip(vec, group_size=32):
        """Quantize+dequantize a vector; return (recovered, mean_sq_err)."""
        out = []
        errs = []
        for i in range(0, len(vec), group_size):
            g = vec[i:i + group_size]
            q, scale = INT4.quantize_group(g)
            dq = INT4.dequantize(q, scale)
            out.extend(dq)
            errs.append(sum((a - b) ** 2 for a, b in zip(g, dq)) / len(g))
        return out, sum(errs) / max(len(errs), 1)
