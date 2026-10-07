"""chonkforge example.

Run from anywhere:  python examples/client_chat.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Stdlib HTTP client for the OpenAI-compatible endpoint."""
import json
import urllib.request


def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8000" + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    print(post("/health", {}))
    print(post("/v1/chat/completions", {
        "model": "chonkforge-mock",
        "messages": [{"role": "user", "content": "explain native multimodal"}],
    }))
    print(post("/v1/answer", {
        "domain": "coding", "question": "what gives O(1) lookup?",
        "options": ["hash table", "linked list"],
    }))


if __name__ == "__main__":
    main()
