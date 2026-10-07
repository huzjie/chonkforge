"""vLLM backend adapter (lazy import)."""
from .base import BaseBackend
from . import register_backend


@register_backend("vllm")
class VLLMBackend(BaseBackend):
    backend_name = "vllm"

    def answer(self, domain, question, options, image=None):
        from vllm import LLM, SamplingParams  # requires vllm installed
        if not hasattr(self, "_llm"):
            self._llm = LLM(model=self.cfg.backend.model or "mistralai/Mistral-Large-4")
        prompt = question + "\n" + "\n".join(options)
        out = self._llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0.0))
        text = out[0].outputs[0].text
        for i, o in enumerate(options):
            if o.lower() in text.lower():
                return i
        return 0

    def train_step(self, domain, question, options, image=None):
        raise NotImplementedError("vllm backend is inference-only")
