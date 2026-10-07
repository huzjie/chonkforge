"""CPU backend: runs the actual pure-Python transformer."""
from ..model.engine import ChonkEngine
from .base import BaseBackend
from . import register_backend


@register_backend("cpu")
class CpuBackend(BaseBackend):
    backend_name = "cpu"

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.engine = ChonkEngine(self.cfg.model)

    def answer(self, domain, question, options, image=None):
        segments = [("text", question)]
        if image is not None:
            segments.append(("image", self.engine.patch.embed(image)))
        scores = self.engine.score_options(segments, options)
        return scores.index(max(scores))

    def train_step(self, domain, question, options, image=None):
        # CPU reference is static; reuse mock's monotonic skill for the demo curve.
        pred = self.answer(domain, question, options, image)
        from ..model.world import world_answer
        correct = world_answer(domain, question, len(options))
        self.skill = min(1.0, self.skill + 0.05 * (0.5 if pred == correct else 1.0))
        return {"skill": round(self.skill, 4), "correct": pred == correct}
