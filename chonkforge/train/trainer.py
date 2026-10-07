"""End-to-end training orchestrator: pretrain -> SFT -> RL.

The reference implementation uses a single trainable "skill" knob as the
capability parameter (see docs/TRAINING.md for why). Each phase drives it
monotonically toward 1.0 while reporting its own metric:

  - pretrain : normalized next-token loss  (decreases)
  - sft      : instruction-following accuracy (increases)
  - rl       : GRPO/PPO reward             (increases)
"""
from .pretrain import Pretrainer
from .sft import SFTTrainer
from .rl import RLTrainer


class Trainer:
    def __init__(self, backend, cfg=None):
        self.backend = backend
        self.cfg = cfg

    def pretrain(self, steps=20, seq_len=8):
        return Pretrainer(self.backend).run(steps=steps, seq_len=seq_len)

    def sft(self, steps=30, n_domains=6, n_q=8):
        return SFTTrainer(self.backend).run(steps=steps, n_domains=n_domains, n_q=n_q)

    def rl(self, steps=30, group_size=4):
        return RLTrainer(self.backend).run(steps=steps, group_size=group_size)

    def run_pipeline(self, **kw):
        pretrain = self.pretrain(kw.get("pretrain_steps", 20))
        sft = self.sft(kw.get("sft_steps", 30))
        rl = self.rl(kw.get("rl_steps", 30))
        return {"pretrain": pretrain, "sft": sft, "rl": rl}


def run_pipeline(backend, **kw):
    return Trainer(backend).run_pipeline(**kw)
