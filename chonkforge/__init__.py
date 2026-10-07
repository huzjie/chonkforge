"""chonkforge: a natively-multimodal 1T sparse-MoE frontier framework.

A zero-dependency, fully runnable reference implementation of the engineering
ideas behind Mistral Large 4 "Le Chonk" (announced 2026-10-07): a ~1T-parameter
Mixture-of-Experts model with ~49B active parameters, a 1M-token context window,
trained on ~3,800 NVIDIA Grace Blackwell GPUs in European sovereign data centres,
and natively multimodal -- a *single* transformer backbone processes interleaved
text and vision tokens together, with no bolted-on frozen vision encoder.

This repo distills its most reusable engineering ideas into one framework:

1. Native multimodal fusion -- text tokens and image-patch tokens share one
   embedding space and one MoE backbone (no separate encoder + adapter).
2. Extreme sparse MoE -- top-k routing, SwiGLU experts, load-balancing loss,
   ~20x sparse ratio (1T total / 49B active).
3. 1M-context attention -- interleaved sliding-window local attention + a small
   set of global tokens (linear-ish cost, long-context retrieval).
4. Multi-domain capability heads -- coding / agentic / cybersecurity / finance /
   law / visual grounding, each a runnable benchmark.
5. Sovereign compute governance -- region / jurisdiction / data-residency tags
   and an audit log (the European-sovereignty angle).

Highlights
----------
- Zero-dependency pure-Python tensor core with autodiff over the op set
- Deterministic trainable mock backend (skill monotonically -> 1.0)
- Six-domain benchmark suite + FP8/INT4 quantization simulation
- OpenAI-compatible stdlib HTTP serving + CLI + Docker/K8s/Helm/CI
- Prompt library + recipes + extensive docs
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
