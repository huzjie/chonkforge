"""Benchmark base class."""
from ..model.world import world_answer


class BaseBenchmark:
    domain = "base"
    n_tasks = 16

    def tasks(self):
        """Return a list of (question, options, image_or_None)."""
        raise NotImplementedError

    def run(self, backend):
        correct = 0
        total = 0
        per = []
        for question, options, image in self.tasks():
            pred = backend.answer(self.domain, question, options, image)
            target = world_answer(self.domain, question, len(options))
            correct += 1 if pred == target else 0
            total += 1
            per.append({"question": question, "pred": pred, "target": target})
        return {
            "domain": self.domain,
            "accuracy": round(correct / max(total, 1), 4),
            "n": total,
            "correct": correct,
            "per": per,
        }
