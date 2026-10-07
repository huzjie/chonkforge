"""serve: start the stdlib HTTP server."""
from ..backends import get_backend


def serve(cfg, host=None, port=None):
    backend = get_backend(cfg.backend.name, cfg)
    from ..serving import run_server
    run_server(backend, host=host or cfg.serve.host, port=port or cfg.serve.port,
               auth_token=cfg.serve.auth_token)
