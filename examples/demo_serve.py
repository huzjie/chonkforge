"""chonkforge example.

Run from anywhere:  python examples/demo_serve.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Start the HTTP server and hit it with the stdlib client."""
from chonkforge.config import load_config
from chonkforge.backends import get_backend


def main():
    cfg = load_config()
    backend = get_backend("mock", cfg)
    from chonkforge.serving import run_server
    run_server(backend, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
