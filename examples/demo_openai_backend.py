"""chonkforge example.

Run from anywhere:  python examples/demo_openai_backend.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Show how to point the openai backend at any OpenAI-compatible endpoint."""
from chonkforge.config import Config
from chonkforge.backends import get_backend


def main():
    cfg = Config()
    cfg.backend.name = "openai"
    cfg.backend.base_url = "https://api.openai.com/v1"
    cfg.backend.api_key = "sk-..."
    cfg.backend.model = "gpt-4o-mini"
    backend = get_backend("openai", cfg)
    # backend.answer("coding", "O(1) lookup?", ["hash table", "linked list"])
    print("openai backend ready (set api_key to enable calls)")


if __name__ == "__main__":
    main()
