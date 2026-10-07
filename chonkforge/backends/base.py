"""Backend base class."""
from ..config import Config


class BaseBackend:
    backend_name = "base"

    def __init__(self, cfg=None, **kw):
        self.cfg = cfg or Config()
        self.skill = 0.5

    def answer(self, domain, question, options, image=None):
        raise NotImplementedError

    def train_step(self, domain, question, options, image=None):
        raise NotImplementedError

    def get_skill(self):
        return getattr(self, "skill", 0.5)
