"""train: pretrain -> sft -> rl pipeline demo."""
from ..backends import get_backend


def train(cfg, steps=30):
    backend = get_backend(cfg.backend.name, cfg)
    from ..train import Trainer
    t = Trainer(backend)
    print("== pretrain (loss down) ==")
    for step, skill, loss in t.pretrain(steps=20):
        print("  step %02d skill=%.3f loss=%.4f" % (step, skill, loss))
    print("== sft (accuracy up) ==")
    for step, skill, acc in t.sft(steps=steps):
        print("  step %02d skill=%.3f acc=%.4f" % (step, skill, acc))
    print("== rl (reward up) ==")
    for step, skill, reward in t.rl(steps=steps):
        print("  step %02d skill=%.3f reward=%.4f" % (step, skill, reward))
    print("final skill = %.3f" % backend.get_skill())
