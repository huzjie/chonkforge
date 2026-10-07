"""CLI entry: doctor / train / bench / serve / quantize."""
import argparse

from ..config import load_config


def _common(parser):
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument("--backend", default=None, help="backend name (default from config)")


def build_parser():
    p = argparse.ArgumentParser(prog="chonkforge", description="chonkforge CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment + component self-check")
    _common(d)

    t = sub.add_parser("train", help="pretrain -> sft -> rl pipeline")
    _common(t)
    t.add_argument("--steps", type=int, default=30)

    b = sub.add_parser("bench", help="run the 6-domain benchmark suite")
    _common(b)

    s = sub.add_parser("serve", help="start the stdlib HTTP server")
    _common(s)
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)

    q = sub.add_parser("quantize", help="FP8/INT4 quantization demo")
    _common(q)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.backend.name = args.backend

    if args.cmd == "doctor":
        from .doctor import doctor
        doctor(cfg)
    elif args.cmd == "train":
        from .train import train
        train(cfg, steps=args.steps)
    elif args.cmd == "bench":
        from .bench import bench
        bench(cfg)
    elif args.cmd == "serve":
        from .serve import serve
        serve(cfg, host=args.host, port=args.port)
    elif args.cmd == "quantize":
        from .quantize import quantize
        quantize(cfg)
