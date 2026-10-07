"""SFT phase: supervised instruction-following on multi-domain QA pairs."""
from ..model.world import world_answer
from ..utils.stable import stable_float

DOMAINS = ["coding", "agentic", "cybersecurity", "finance", "law", "visual_grounding"]
OPTIONS = ["A", "B", "C", "D"]


class SFTTrainer:
    def __init__(self, backend):
        self.backend = backend

    def run(self, steps=30, n_domains=6, n_q=8):
        hist = []
        correct = 0
        total = 0
        for step in range(steps):
            d = DOMAINS[step % len(DOMAINS)]
            q = "q%d" % (step % n_q)
            options = OPTIONS[:2 + (step % 3)]  # 2..4 options
            res = self.backend.train_step(d, q, options)
            total += 1
            correct += 1 if res["correct"] else 0
            hist.append((step, round(self.backend.get_skill(), 4), round(correct / total, 4)))
        return hist
