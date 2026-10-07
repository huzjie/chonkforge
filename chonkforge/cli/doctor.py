"""doctor: environment + component self-check."""
from ..config import Config
from ..backends import list_backends, get_backend


def doctor(cfg):
    print("=" * 60)
    print("chonkforge doctor")
    print("=" * 60)
    print("[cfg] model: dim=%d hidden=%d n_experts=%d top_k=%d layers=%d"
          % (cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
             cfg.model.top_k, cfg.model.n_layers))
    print("[cfg] scale hint: %s total / %s active (see docs/SCALING.md)"
          % (cfg.model.total_params_hint, cfg.model.active_params_hint))
    print("[cfg] backend: %s" % cfg.backend.name)

    print("[backends] available = %s" % ", ".join(list_backends()))
    backend = get_backend(cfg.backend.name, cfg)
    print("[backend] %s skill=%.2f" % (backend.backend_name, backend.get_skill()))

    # exercise the native-multimodal forward (small config -> fast)
    from ..model.engine import ChonkEngine
    engine = ChonkEngine(cfg.model)
    # text-only forward
    seg = engine.fusion.build([("text", "hello chonk")])
    hidden = engine.transformer.forward(seg.embeddings)
    pooled = engine.transformer.pooled(hidden)
    print("[mm] text sequence tokens=%d dim=%d pooled[0]=%.4f"
          % (len(seg), len(pooled), pooled[0]))

    # image forward (native multimodal path)
    from ..mm.vision import make_image
    img = make_image(cfg.model.img_h, cfg.model.img_w, key="doctor.img")
    tok = engine.patch.embed(img)
    seg2 = engine.fusion.build([("text", "what is this?"), ("image", tok)])
    hidden2 = engine.transformer.forward(seg2.embeddings)
    pooled2 = engine.transformer.pooled(hidden2)
    print("[mm] multimodal sequence tokens=%d (visual=%d) pooled[0]=%.4f"
          % (len(seg2), len(tok.patch_ids), pooled2[0]))

    # sparse MoE stats
    from ..moe.layer import MoELayer
    layer = MoELayer(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                     cfg.model.top_k, seed=1)
    x = [0.01 * (i % 5) for i in range(cfg.model.dim)]
    _, stats = layer.forward(x)
    print("[moe] n_experts=%d top_k=%d sparse_ratio=%.1fx aux_loss=%.4f"
          % (stats.n_experts, stats.top_k, stats.sparse_ratio, stats.aux_loss))

    print("[doctor] OK")
