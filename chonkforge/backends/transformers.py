"""HuggingFace transformers backend adapter (lazy import)."""
from .base import BaseBackend
from . import register_backend


@register_backend("transformers")
class TransformersBackend(BaseBackend):
    backend_name = "transformers"

    def answer(self, domain, question, options, image=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        if not hasattr(self, "_tok"):
            self._tok = AutoTokenizer.from_pretrained(
                self.cfg.backend.model or "mistralai/Mistral-Large-4")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.backend.model or "mistralai/Mistral-Large-4")
        ids = self._tok(question, return_tensors="pt")
        out = self._model.generate(**ids, max_new_tokens=16)
        text = self._tok.decode(out[0], skip_special_tokens=True)
        for i, o in enumerate(options):
            if o.lower() in text.lower():
                return i
        return 0

    def train_step(self, domain, question, options, image=None):
        raise NotImplementedError("transformers backend is inference-only")
