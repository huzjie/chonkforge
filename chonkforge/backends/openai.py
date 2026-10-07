"""OpenAI-compatible backend (lazy; needs `requests` + an endpoint)."""
from .base import BaseBackend
from . import register_backend


@register_backend("openai")
class OpenAIBackend(BaseBackend):
    backend_name = "openai"

    def answer(self, domain, question, options, image=None):
        return self._call(question, options)

    def _call(self, question, options):
        import requests
        url = (self.cfg.backend.base_url or "https://api.openai.com/v1") + "/chat/completions"
        headers = {"Authorization": "Bearer " + self.cfg.backend.api_key}
        prompt = question + "\n" + "\n".join("%d) %s" % (i, o) for i, o in enumerate(options))
        r = requests.post(url, headers=headers, json={
            "model": self.cfg.backend.model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
        }, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        for i, o in enumerate(options):
            if o.lower() in text.lower():
                return i
        return 0

    def train_step(self, domain, question, options, image=None):
        raise NotImplementedError("openai backend is inference-only")
