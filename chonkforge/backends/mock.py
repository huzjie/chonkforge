"""Deterministic trainable mock backend.

The reusable trick (borrowed from every prior forge project): a mock backend
whose correctness is gated by a single scalar `skill` mixed with deterministic
noise. Because the noise is md5-seeded per (domain, question, option), results
are 100% reproducible, and because `skill` is the only thing that shifts the
correct option's score, training `skill` monotonically -> 1.0 gives a clean,
readable accuracy curve from ~1/N to 1.0.
"""
from ..utils.stable import stable_float
from ..model.world import world_answer
from .base import BaseBackend
from . import register_backend


@register_backend("mock")
class MockBackend(BaseBackend):
    backend_name = "mock"

    def answer(self, domain, question, options, image=None):
        correct = world_answer(domain, question, len(options))
        scores = []
        for i in range(len(options)):
            noise = stable_float("mock.score:%s:%s:%d" % (domain, question, i), 0.0, 1.0)
            scores.append(self.skill * (1.0 if i == correct else 0.0) + noise)
        return scores.index(max(scores))

    def train_step(self, domain, question, options, image=None):
        pred = self.answer(domain, question, options, image)
        correct = world_answer(domain, question, len(options))
        self.skill = min(1.0, self.skill + 0.05 * (0.5 if pred == correct else 1.0))
        return {"skill": round(self.skill, 4), "correct": pred == correct}
