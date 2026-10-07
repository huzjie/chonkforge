"""Pretraining phase: normalized next-token loss decreases as skill grows."""
import math


class Pretrainer:
    def __init__(self, backend):
        self.backend = backend

    def run(self, steps=20, seq_len=8):
        hist = []
        for step in range(steps):
            # a synthetic next-token objective: normalized loss tied to capability
            loss = 1.0 - self.backend.get_skill()
            # capability grows monotonically (the "learned" quantity)
            self._grow()
            hist.append((step, round(self.backend.get_skill(), 4), round(loss, 4)))
        return hist

    def _grow(self):
        s = self.backend.get_skill()
        s = min(1.0, s + 0.05 * (1.0 - s) + 0.005)
        if hasattr(self.backend, "skill"):
            self.backend.skill = s
        elif hasattr(self.backend, "engine") and hasattr(self.backend.engine, "skill"):
            self.backend.engine.skill = s
