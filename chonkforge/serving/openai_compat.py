"""OpenAI-compatible request/response shapes over a stdlib server."""
import json
import time
from ..utils.stable import stable_float


class OpenAICompatHandler:
    def __init__(self, backend, auth_token=""):
        self.backend = backend
        self.auth_token = auth_token

    def _authed(self, headers):
        if not self.auth_token:
            return True
        return headers.get("Authorization") == "Bearer " + self.auth_token

    def chat_completion(self, body):
        messages = body.get("messages", [])
        question = messages[-1]["content"] if messages else ""
        # deterministic pseudo-completion gated by skill
        choices = int(round(stable_float("srv.choices", 0, 3)))
        content = ("Reasoned: %s | confidence=%.2f | skill=%.2f"
                   % (question[:60], self.backend.get_skill(), self.backend.get_skill()))
        return {
            "id": "chatcmpl-" + str(int(time.time() * 1000)),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.backend.backend_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": len(question), "completion_tokens": 16, "total_tokens": len(question) + 16},
        }

    def models(self):
        return {
            "object": "list",
            "data": [{"id": "chonkforge-" + self.backend.backend_name,
                      "object": "model", "owned_by": "chonkforge"}],
        }

    def answer(self, body):
        domain = body.get("domain", "coding")
        question = body.get("question", "")
        options = body.get("options", ["A", "B"])
        idx = self.backend.answer(domain, question, options)
        return {"domain": domain, "question": question, "answer": idx,
                "option": options[idx] if 0 <= idx < len(options) else None,
                "skill": self.backend.get_skill()}
