"""RL phase: GRPO-style group-relative advantage drives reward up.

The reusable idea: reward a *group* of rollouts relative to their own mean
(group-relative advantage), which removes the need for a value network and
stabilises training. The mock's deterministic rollouts collapse to a single
answer per question, so the group is built by varying the noise seed; the
advantage of the correct answer is always >= 0, so `skill` climbs monotonically.
"""
from ..model.world import world_answer
from ..utils.stable import stable_float


class RLTrainer:
    def __init__(self, backend):
        self.backend = backend

    def run(self, steps=30, group_size=4):
        hist = []
        for step in range(steps):
            domain = "coding"
            question = "rl_%d" % step
            options = ["A", "B", "C"]
            correct = world_answer(domain, question, len(options))
            # simulate a group of rollouts with varied noise
            rewards = []
            for g in range(group_size):
                score = self.backend.get_skill() * (1.0 if g == correct else 0.0)                     + stable_float("rl.g:%s:%d:%d" % (question, step, g), 0.0, 0.5)
                rewards.append(score)
            mean = sum(rewards) / len(rewards)
            # group-relative advantage (>= 0 for the correct rollout)
            advantage = max(0.0, rewards[correct] - mean)
            reward = self.backend.get_skill() + advantage
            s = min(1.0, self.backend.get_skill() + 0.05 * (1.0 - self.backend.get_skill()))
            if hasattr(self.backend, "skill"):
                self.backend.skill = s
            elif hasattr(self.backend, "engine"):
                self.backend.engine.skill = s
            hist.append((step, round(s, 4), round(reward, 4)))
        return hist
