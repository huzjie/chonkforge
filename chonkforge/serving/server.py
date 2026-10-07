"""Zero-dependency stdlib HTTP server (OpenAI-compatible subset)."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .openai_compat import OpenAICompatHandler


class Handler(BaseHTTPRequestHandler):
    app = None  # set to OpenAICompatHandler before serving

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "skill": self.app.backend.get_skill()})
        elif self.path == "/v1/models":
            self._send(200, self.app.models())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.app._authed(self.headers):
            self._send(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            self._send(400, {"error": "bad json"})
            return
        if self.path == "/v1/chat/completions":
            self._send(200, self.app.chat_completion(body))
        elif self.path == "/v1/answer":
            self._send(200, self.app.answer(body))
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # quiet


def make_server(backend, auth_token=""):
    return OpenAICompatHandler(backend, auth_token)


def run_server(backend, host="127.0.0.1", port=8000, auth_token=""):
    app = make_server(backend, auth_token)
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("chonkforge serving on http://%s:%d" % (host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
