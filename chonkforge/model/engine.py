"""ChonkEngine: high-level model facade (multimodal in, answer out)."""
from ..mm.tokenizer import TextTokenizer
from ..mm.vision import PatchEmbedder
from ..mm.fusion import NativeFusion
from .transformer import make_transformer
from .world import world_answer


class ChonkEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = TextTokenizer()
        self.patch = PatchEmbedder(cfg.dim, cfg.patch, cfg.img_h, cfg.img_w)
        self.transformer = make_transformer(cfg)
        self.fusion = NativeFusion(self.transformer.emb.lookup, self.patch, self.tokenizer)
        self.skill = 0.5  # trainable capability knob

    def embed_segments(self, segments):
        seq = self.fusion.build(segments)
        return seq

    def pooled(self, segments):
        seq = self.embed_segments(segments)
        hidden = self.transformer.forward(seq.embeddings)
        return self.transformer.pooled(hidden)

    def score_options(self, segments, options):
        """Pool the multimodal sequence, score each option by embedding similarity."""
        ctx = self.pooled(segments)
        scores = []
        for opt in options:
            oe = self.transformer.emb.lookup(abs(hash(opt)) % self.cfg.vocab)
            scores.append(sum(ctx[j] * oe[j] for j in range(len(ctx))))
        return scores

    def answer(self, domain, question, options, image=None):
        """Deterministic mock-style answer gated by skill."""
        correct = world_answer(domain, question, len(options))
        from ..utils.stable import stable_float
        scores = []
        for i in range(len(options)):
            noise = stable_float("eng.score:%s:%s:%d" % (domain, question, i), 0.0, 1.0)
            scores.append(self.skill * (1.0 if i == correct else 0.0) + noise)
        return scores.index(max(scores)), correct

    def train_step(self, domain, question, options, image=None):
        pred, correct = self.answer(domain, question, options, image)
        # always-positive delta => skill monotonically -> 1.0
        self.skill = min(1.0, self.skill + 0.05 * (0.5 if pred == correct else 1.0))
        return {"skill": round(self.skill, 4), "correct": pred == correct}
